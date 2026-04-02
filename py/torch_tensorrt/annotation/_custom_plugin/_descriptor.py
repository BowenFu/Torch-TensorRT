"""tta.custom_plugin: CustomPluginSpec, factory, and QDP registration.

This module provides:
- CustomPluginSpec: the descriptor returned by tta.custom_plugin(...)
- custom_plugin(): factory that computes a deterministic QDP op_name
- register_custom_plugin(): registers @trtp.register / @trtp.autotune / @trtp.aot_impl
- lower_custom_plugin_descriptor(): lowers a CustomPluginSpec to a
  TRT plugin layer via trtp.op

Backend-specific AOT logic lives in _triton_aot / _cutile_aot /
_cutedsl_aot.
"""
from __future__ import annotations

import inspect
import logging
import threading
import typing

import torch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
    trt = None  # type: ignore[assignment]
    trtp = None  # type: ignore[assignment]

from ._qdp_utils import (
    QDPRuntimeError,
    TacticEntry,
    build_tactic_table,
    derive_impl_id,
    make_meta_tensor_from_td,
    make_meta_tensor_from_td_symbolic,
    make_qdp_symbol,
    make_td_from_meta_using_template,
    make_td_from_meta_using_template_symbolic,
)
from .._layer_metadata import set_tta_layer_metadata
from .._specs import CuTeDSLSpec, CuTileSpec, TritonSpec
from ._symbolic import SymbolicTensor, TensorRole

KernelSpec = Union[TritonSpec, CuTileSpec, CuTeDSLSpec]

# ---------------------------------------------------------------------------
# Process-level QDP registration registry
# ---------------------------------------------------------------------------
# TRT's internal QDP registry is process-global, so we track registered op_names
# at the process level rather than using thread-local storage.  Thread-local storage
# would allow two concurrent threads (e.g. pytest-xdist workers sharing a process)
# to each attempt QDP registration for the same op_name, causing TRT to raise an
# "already registered" error on the second attempt.
#
# Threading contract for _qdp_registered_ops:
#   WRITE: Always done under _qdp_registration_lock.  The set is only grown, never
#          shrunk, so writes are monotonic.  The final add() and the return from
#          register_custom_plugin() are both inside the lock.
#   READ (under lock): Always safe; used inside register_custom_plugin() after
#          acquiring _qdp_registration_lock for the definitive TOCTOU-safe check.
#   READ (without lock, fast path): Safe because the set only grows.  A thread
#          that observes op_name IN the set can safely skip registration — the
#          worst outcome of a race is a redundant lock acquisition on the slow
#          path, which is also safe.  A thread that observes op_name NOT IN the
#          set proceeds to acquire the lock and re-checks inside (double-checked
#          locking pattern).  This avoids lock contention on the common post-
#          registration path without risking double-registration.
_qdp_registered_ops: set = set()
_qdp_registration_lock = threading.Lock()

# Thread-local cache for _aot_fn builders.  Building the closure is cheap,
# but we avoid redundant work within a single thread.
_tls = threading.local()


def _get_aot_fn_cache() -> Dict[Tuple[str, int], Callable[..., Any]]:
    """Return the thread-local cache mapping (op_name, num_inputs) to built _aot_fn.

    The cache is never cleared because TRT's QDP registry is also process-persistent,
    so the two remain in sync without any explicit eviction.
    """
    if not hasattr(_tls, "aot_fn_cache"):
        _tls.aot_fn_cache = {}
    return _tls.aot_fn_cache


# ---------------------------------------------------------------------------
# Public descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomPluginSpec:
    """Descriptor returned by ``tta.custom_plugin(...)``.

    Lifecycle
    ---------
    1. **Creation** — ``custom_plugin()`` constructs a ``CustomPluginSpec``
       by splitting kwargs into weights/attrs, hashing the kernel specs
       to derive a deterministic ``op_name``, and probing ``meta_impl`` with dummy
       meta tensors to count outputs.  No TRT objects are touched at this stage.

    2. **QDP registration** — ``register_custom_plugin()`` (called lazily from
       ``lower_custom_plugin_descriptor()``) registers three QDP callbacks with
       TRT's process-global plugin registry under ``op_name``:

       * ``@trtp.register``   — shape/dtype descriptor (uses ``meta_impl`` or identity)
       * ``@trtp.autotune``   — enumerates (dtype, format, tactic) combinations
       * ``@trtp.aot_impl``   — AOT kernel dispatch (Triton / CuTile / CuTeDSL)

       Registration is idempotent: a process-level set (``_qdp_registered_ops``)
       guards against double-registration across threads.

    3. **Use in lowering** — ``lower_custom_plugin_descriptor()`` calls
       ``ctx.net.add_plugin(trtp.op.<ns>.<name>(*trt_inputs), aot=True)`` to add
       the plugin layer to the TRT network.  Weight tensors declared in kwargs are
       injected as ``trt.add_constant`` layers and appended to ``trt_inputs`` before
       the plugin call so the launch_fn receives
       ``(*activations, *weights_in_order, *outputs, ...)``.

    Attributes:
        op_name:     Unique QDP symbol, e.g. ``"tta_custom::host_kernel_a1b2c3d4"``.
                     Deterministically derived from the kernel specs and attrs so that
                     re-creating the same descriptor in a different process produces
                     the same name.
        specs:       Non-empty list of kernel specs (TritonSpec, CuTileSpec,
                     CuTeDSLSpec).  Multiple specs provide alternative tactics; TRT's
                     autotune selects the fastest one at engine-build time.
        meta_impl:   PyTorch meta function for shape/dtype inference.  Receives meta
                     tensors (one per plugin input) and must return a single
                     ``torch.Tensor`` or a tuple/list of ``torch.Tensor`` s.
        num_outputs: Number of output tensors, inferred from ``meta_impl`` at
                     creation time (defaults to 1 when ``meta_impl`` is ``None``).
        attrs:       Scalar kwargs baked into kernel PTX at AOT time
                     (e.g. ``addend=1.0``).  NOT forwarded as TRT plugin fields.
        weights:     Tensor kwargs bound at plugin creation time.  At lowering, each
                     weight is added to the TRT network as a constant tensor and
                     appended to the dynamic activation inputs.  The launch_fn must
                     accept dynamic inputs first, then weights (in declaration order),
                     then outputs.  Named so debug messages can identify each weight.
    """

    op_name: str
    specs: List[KernelSpec]
    meta_impl: Callable[..., Any]
    num_outputs: int = 1
    attrs: Dict[str, Any] = field(default_factory=dict)
    # Tensor weights are excluded from __hash__ and __eq__ because torch.Tensor
    # is not hashable.  Identity / equality of a descriptor is captured by
    # op_name (which already encodes num_weights via derive_impl_id).
    weights: Dict[str, "torch.Tensor"] = field(
        default_factory=dict, hash=False, compare=False
    )

    def register_dynamo_plugin(
        self,
        op_name: str,
        capability_validator: Optional[Any] = None,
        priority: Any = None,
        supports_dynamic_shapes: bool = False,
        requires_output_allocator: bool = False,
    ) -> None:
        """Register this spec as a QDP plugin + Dynamo converter for ``op_name``.

        Full registration in three steps:

        1. ``auto_register_torch_op`` — registers ``torch.library.custom_op`` and
           ``register_fake`` from ``launch_fn`` / ``meta_impl``.
        2. ``register_qdp`` — registers ``@trtp.register`` / ``@trtp.autotune`` /
           ``@trtp.aot_impl`` under the torch op name.
        3. A Dynamo converter is registered that calls :meth:`lower_to_trt`,
           routing through :func:`lower_custom_plugin_descriptor` to share
           weight-injection, TTA metadata, and ``aot=True`` with the native path.

        Dynamo utilities (``dynamo_tensorrt_converter``, ``get_trt_tensor``) are
        imported locally inside this method to avoid a circular dependency between
        the annotation module and ``torch_tensorrt.dynamo``.

        Args:
            op_name:                  torch op name in ``"namespace::name"`` form.
            capability_validator:     optional node capability predicate.
            priority:                 converter registry priority.
            supports_dynamic_shapes:  whether the converter supports dynamic shapes.
            requires_output_allocator: whether the converter requires an output allocator.
        """
        import uuid
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            ConverterPriority,
            dynamo_tensorrt_converter,
        )
        from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

        if priority is None:
            priority = ConverterPriority.STANDARD

        self.auto_register_torch_op(op_name)

        namespace, name = op_name.split("::")
        torch_op = torch.ops
        for part in (namespace, name):
            torch_op = getattr(torch_op, part)
        schema = torch_op._schemas[""]
        tensor_input_names = [
            arg.name for arg in schema.arguments
            if arg.type.isSubtypeOf(torch._C.TensorType.get())
        ]
        # Weights are injected as trt.add_constant layers and appended to the
        # dynamic activation inputs before the plugin sees them, so the total
        # QDP input count is dynamic_inputs + weights.
        self.register_qdp(op_name, len(tensor_input_names) + len(self.weights))

        torch_overload = getattr(torch_op, "default")
        _impl = self
        _qdp_name = op_name
        _tensor_input_names = tensor_input_names

        def _tta_converter(ctx: Any, target: Any, args: Any, kwargs: Any, name: str) -> Any:
            unique_id = uuid.uuid4()
            itensor_args = [
                get_trt_tensor(ctx, t, f"{t_name}_{unique_id}")
                for t, t_name in zip(args[: len(_tensor_input_names)], _tensor_input_names)
            ]
            return _impl.lower_to_trt(ctx, itensor_args, name, qdp_name=_qdp_name)

        dynamo_tensorrt_converter(
            torch_overload,
            capability_validator=capability_validator,
            priority=priority,
            supports_dynamic_shapes=supports_dynamic_shapes,
            requires_output_allocator=requires_output_allocator,
        )(_tta_converter)

    def lower_to_trt(
        self,
        ctx: Any,
        trt_inputs: List[Any],
        name: str,
        qdp_name: Optional[str] = None,
    ) -> Any:
        """Lower this spec to a TRT ``IPluginV3`` layer via :func:`lower_custom_plugin_descriptor`.

        Shared entry-point used by both the native TTA lowering pass and the
        Dynamo integration path (``trt_plugins.custom_op(impl=...)``).  Using
        this method ensures both paths go through the same code: weight
        injection, TTA layer metadata, and ``aot=True`` semantics.

        When ``qdp_name`` is supplied the plugin is looked up under that name
        (the torch op name, e.g. ``"ns::my_op"``) rather than the auto-derived
        TTA fingerprint in ``self.op_name``.  This is required when the plugin
        was registered via :meth:`register_qdp` which registers under the torch
        op name rather than the TTA fingerprint.

        Args:
            ctx:        Torch-TRT ``ConversionContext`` (carries ``ctx.net``).
            trt_inputs: Ordered ``trt.ITensor`` activation inputs.
            name:       Layer name for TRT debugging/profiling.
            qdp_name:   Optional QDP name override (e.g. torch op name).

        Returns:
            A single ``trt.ITensor`` or a tuple of ``trt.ITensor`` s.
        """
        import dataclasses
        desc = dataclasses.replace(self, op_name=qdp_name) if qdp_name is not None else self
        return lower_custom_plugin_descriptor(ctx, desc, trt_inputs, name)

    def register_qdp(self, qdp_name: str, num_inputs: int) -> None:
        """Register this spec as a QDP plugin under ``qdp_name``.

        Thin wrapper around :func:`register_custom_plugin` that lets callers
        in other packages (e.g. ``torch_tensorrt.dynamo.conversion.plugins``)
        trigger QDP registration without importing the ``annotation`` module.

        Args:
            qdp_name:   TRT QDP op name to register under (e.g. ``"ns::op"``).
            num_inputs: Number of tensor inputs the plugin accepts.
        """
        register_custom_plugin(self, num_inputs=num_inputs, num_outputs=self.num_outputs, qdp_name=qdp_name)

    def auto_register_torch_op(self, op_name: str) -> None:
        """Auto-register ``torch.library.custom_op`` and ``register_fake`` for ``op_name``.

        Eliminates the boilerplate of writing ``@torch.library.custom_op`` and
        ``@torch.library.register_fake`` by hand when a :class:`CustomPluginSpec`
        already carries ``meta_impl`` and kernel specs.

        The eager implementation calls the first spec's ``launch_fn`` with the
        first available config.  The fake implementation delegates to
        ``meta_impl`` for shape/dtype inference.

        If the op is already registered in ``torch.ops``, the call is a no-op.

        Args:
            op_name: torch op name in ``"namespace::name"`` form.

        Raises:
            ValueError: If ``meta_impl`` is ``None`` (required for shape inference).
        """
        import inspect

        if self.meta_impl is None:
            raise ValueError(
                f"auto_register_torch_op: meta_impl is required to auto-register "
                f"'{op_name}'; set meta_impl in tta.custom_plugin(..., meta_impl=...)"
            )

        namespace, name = op_name.split("::")

        # Skip if the op is already registered.
        ns_obj = getattr(torch.ops, namespace, None)
        if ns_obj is not None and hasattr(ns_obj, name):
            return

        meta_sig = inspect.signature(self.meta_impl)
        param_names = list(meta_sig.parameters.keys())

        first_spec = self.specs[0]
        first_config = first_spec.configs[0] if getattr(first_spec, "configs", None) else {}
        _launch = first_spec.launch_fn
        _meta = self.meta_impl
        _num_outputs = self.num_outputs

        # CuTeDSL @cute.jit functions expect cute.Tensor, not torch.Tensor.
        # Wrap the launch to convert inputs via from_dlpack before calling.
        if isinstance(first_spec, CuTeDSLSpec):
            _jit_launch = _launch

            def _launch(*args, **kwargs):
                try:
                    from cutlass.cute.runtime import from_dlpack as _from_dlpack
                except ImportError:
                    raise
                # Ensure contiguous memory before DLPack conversion: TRT may
                # pass tensors with non-unit row strides (e.g. from a preceding
                # matmul whose output is padded for alignment).
                converted = [_from_dlpack(a.contiguous()) if isinstance(a, torch.Tensor) else a for a in args]
                return _jit_launch(*converted, **kwargs)

        def _eager_body(*args: torch.Tensor) -> Any:
            meta_outs = _meta(*args)
            if not isinstance(meta_outs, (tuple, list)):
                meta_outs = (meta_outs,)
            outs = [
                torch.empty(o.shape, dtype=o.dtype, device=args[0].device)
                for o in meta_outs
            ]
            _launch(*args, *outs, **first_config)
            return outs[0] if _num_outputs == 1 else list(outs)

        # Reuse the same __signature__ trick as _build_desc_fn: attach a custom
        # inspect.Signature so torch.library's schema inference sees real named
        # Tensor parameters without exec()-generated source code.
        # Multi-output ops use List[torch.Tensor] (→ schema "Tensor[]") so that
        # torch.library registers the correct return type.
        sig_params = [
            inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=torch.Tensor)
            for p in param_names
        ]
        ret_annotation = torch.Tensor if _num_outputs == 1 else List[torch.Tensor]
        sig = inspect.Signature(sig_params, return_annotation=ret_annotation)
        _eager_body.__signature__ = sig
        torch.library.custom_op(op_name, mutates_args=())(_eager_body)

        # Fake impl: forward to meta_impl for shape/dtype inference during tracing.
        # For multi-output ops, return a list to match the "Tensor[]" schema.
        def _fake_body(*args: torch.Tensor) -> Any:
            result = _meta(*args)
            if _num_outputs > 1 and isinstance(result, tuple):
                return list(result)
            return result

        _fake_body.__signature__ = sig
        torch.library.register_fake(op_name)(_fake_body)


# LIMITATION: _infer_num_outputs probes meta_impl with rank-1 dummy tensors to count
# outputs.  It returns 1 (silently) when:
#   (a) a dispatch mode (e.g. FakeTensorMode during torch.export) is active, or
#   (b) meta_impl raises with rank-1 dummies (shape-sensitive implementations).
# In either case a multi-output plugin will be silently treated as single-output,
# causing a downstream shape mismatch at TRT engine build time.  Users with
# shape-sensitive or rank-sensitive meta_impl functions must pass num_outputs
# explicitly to custom_plugin() to avoid this.
def _infer_num_outputs(meta_impl: Callable[..., Any]) -> int:
    """Infer the number of outputs from ``meta_impl`` by calling it with 1-D dummy tensors.

    This must be called outside any active torch dispatch mode (e.g., FakeTensorMode
    during ``torch.export`` tracing): tensor ops inside an active dispatch mode get
    recorded into the FX graph as dead code and break ``run_decompositions``.

    Returns 1 if any dispatch mode is active, or if ``meta_impl`` raises with 1-D
    dummies (e.g. shape-sensitive meta_impls).

    Args:
        meta_impl: Meta function to probe.

    Returns:
        Inferred output count; always >= 1.
    """
    # Skip inference if a torch dispatch mode is active (FakeTensorMode, etc.).
    # Tensor ops created here would be recorded as dead code in the FX graph.
    try:
        if torch._C._len_torch_dispatch_stack() > 0:
            return 1
    except AttributeError:
        # _len_torch_dispatch_stack is a private C++ binding; if it disappears in
        # a future PyTorch version, fail open and continue with the probe.
        pass
    try:
        n_params = len(inspect.signature(meta_impl).parameters)
        dummy = torch.empty(1, device="meta")
        result = meta_impl(*([dummy] * max(n_params, 1)))
        return len(result) if isinstance(result, (tuple, list)) else 1
    except Exception:  # noqa: BLE001
        # meta_impl may be shape-sensitive and raise with 1-D dummies (e.g. a
        # reshape that requires at least 2-D input).  Fall back to 1 output; the
        # caller can always pass num_outputs explicitly via register_custom_plugin.
        return 1


def custom_plugin(
    kernel: Union[KernelSpec, List[KernelSpec]],
    meta_impl: Callable[..., Any],
    **kwargs: Any,
) -> CustomPluginSpec:
    """Create a :class:`CustomPluginSpec` for one or more kernel specs.

    This is the primary entry point for the ``tta.custom_plugin`` API.  The
    returned descriptor is used as the ``impl=`` argument of ``@tta.export_as``.

    Args:
        kernel:    Single kernel spec or a non-empty list of specs.  Multiple specs
                   provide alternative tactics; TRT's autotune benchmarks all of them
                   at engine-build time and selects the fastest.
        meta_impl: Required.  PyTorch meta function used by QDP for shape/dtype
                   inference.  Receives meta tensors (one per plugin input) and
                   must return a single ``torch.Tensor`` or a tuple/list of
                   ``torch.Tensor`` s.  Each tensor's ``.shape`` and ``.dtype``
                   define the output ``TensorDesc`` s.  For the descriptor we use
                   only the first returned tensor; its shape and dtype must match
                   the plugin's actual output so TRT gets correct types and shapes.
        **kwargs:  Plugin-level keyword arguments, split by type at creation time:

                   * ``torch.Tensor`` values → **weights**: frozen tensors bound
                     to this plugin.  At TRT lowering each weight is added to the
                     network as a ``trt.add_constant`` layer and appended to the
                     dynamic activation inputs before calling the plugin.  The
                     launch_fn must accept ``(*activations, *weights,
                     *outputs, ...)``.  The count of weights is included in the
                     ``op_name`` fingerprint so the same kernel spec can be
                     registered with different input arities without stale-
                     registration bugs.

                   * All other values → **attrs**: scalar compile-time constants
                     (e.g. ``addend=1.0``, ``scale=2``).  These are baked into
                     the kernel PTX at AOT time and are NOT forwarded as TRT
                     plugin fields.

    Returns:
        A :class:`CustomPluginSpec` with an auto-computed ``op_name``.

    Raises:
        ValueError: If ``kernel`` is an empty list, or if ``meta_impl`` is ``None``.
        TypeError:  If any element of ``kernel`` is not a valid ``KernelSpec``,
                    or if ``meta_impl`` is not callable.
    """
    specs: List[KernelSpec] = kernel if isinstance(kernel, list) else [kernel]
    if not specs:
        raise ValueError("custom_plugin: kernel list cannot be empty")
    for s in specs:
        if not isinstance(s, (TritonSpec, CuTileSpec, CuTeDSLSpec)):
            raise TypeError(
                f"custom_plugin: kernel must be TritonSpec, CuTileSpec, or "
                f"CuTeDSLSpec, got {type(s).__name__!r}"
            )
    if meta_impl is None:
        raise ValueError(
            "custom_plugin: meta_impl is required and cannot be None; "
            "provide a PyTorch meta function for shape/dtype inference"
        )
    if not callable(meta_impl):
        raise TypeError(
            f"custom_plugin: meta_impl must be callable, "
            f"got {type(meta_impl).__name__!r}"
        )

    # Split kwargs by value type:
    #   torch.Tensor    → weights (TRT constant layers injected at lowering)
    #   everything else → attrs (scalar compile-time constants)
    weights: Dict[str, torch.Tensor] = {}
    attrs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            weights[k] = v
        else:
            attrs[k] = v

    impl_id = derive_impl_id(specs, attrs=attrs, num_weights=len(weights))
    op_name = make_qdp_symbol(impl_id)
    num_outputs = _infer_num_outputs(meta_impl)
    return CustomPluginSpec(
        op_name=op_name, specs=specs, meta_impl=meta_impl,
        num_outputs=num_outputs, attrs=attrs, weights=weights,
    )


# ---------------------------------------------------------------------------
# Shared parameter-list helpers
# ---------------------------------------------------------------------------


def _build_input_params(num_inputs: int, annotation: Any) -> List[inspect.Parameter]:
    """Build a positional ``inspect.Parameter`` list for TRT descriptor functions.

    Args:
        num_inputs:  Number of input parameters to generate (named ``inp0`` … ``inpN``).
        annotation:  Type annotation attached to each parameter (typically
                     ``trtp.TensorDesc``).

    Returns:
        List of ``inspect.Parameter`` objects suitable for use with
        ``inspect.Signature``.
    """
    return [
        inspect.Parameter(f"inp{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
        for i in range(num_inputs)
    ]


def _build_attr_params(attrs: Dict[str, Any]) -> List[inspect.Parameter]:
    """Build an ``inspect.Parameter`` list for QDP plugin field attributes.

    Maps Python value types to the type annotations that TRT's ``@trtp.register``
    validation expects.  Supported primitive types: ``int``, ``float``, ``str``,
    ``bool``.  Any other type falls back to its own ``type()`` as the annotation.

    Args:
        attrs: Mapping of attribute name to scalar value.

    Returns:
        List of ``inspect.Parameter`` objects, one per entry in ``attrs``.
    """
    _type_map = {bool: bool, int: int, float: float, str: str}
    params = []
    for name, value in attrs.items():
        ann = _type_map.get(type(value), type(value))
        params.append(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ann)
        )
    return params


# ---------------------------------------------------------------------------
# Descriptor function builder (shape / dtype via @trtp.register)
# ---------------------------------------------------------------------------


def _build_identity_desc_fn(num_outputs: int) -> Callable[..., Any]:
    """Build a no-``meta_impl`` descriptor that mirrors ``inp0``'s shape/dtype for all outputs.

    Used when ``meta_impl`` is ``None``.  All output ``TensorDesc`` s are produced
    with ``inp0.like()``, giving each output the same dtype, shape, and format as
    the first input.

    Args:
        num_outputs: Number of output tensors to produce.

    Returns:
        A callable suitable for use as the body of a ``@trtp.register`` function.
    """
    _num_outputs = num_outputs

    def _desc(*args: Any) -> Any:  # type: ignore[misc]
        if _num_outputs == 1:
            return args[0].like()
        return tuple(args[0].like() for _ in range(_num_outputs))

    return _desc


def _is_symbolic_shape_expr(shape_expr: Any) -> bool:
    """Return True if any element of shape_expr is a dynamic/symbolic dimension.

    Checks for TRT IDimensionExpr objects (which carry ``is_constant`` or
    ``get_constant_value`` attributes) and falls back to checking whether the
    value is not a plain int/float.  This replaces the fragile
    ``"Fake" in str(type(d))`` heuristic used before TRT 10.x.

    Args:
        shape_expr: Iterable of dimension values from a ``TensorDesc.shape_expr``.

    Returns:
        ``True`` if any element is a TRT symbolic dimension expression; ``False``
        if all elements are concrete Python int/float values.
    """
    for d in shape_expr:
        # Check for TRT IDimensionExpr which are symbolic
        if hasattr(d, 'is_constant') or hasattr(d, 'get_constant_value'):
            return True  # It's a TRT dimension expression
        if not isinstance(d, (int, float)):
            return True
    return False


def _build_meta_impl_desc_fn(
    meta_impl: Callable[..., Any],
    num_outputs: int,
    num_inputs: int,
) -> Callable[..., Any]:
    """Build a ``meta_impl``-based descriptor that infers output ``TensorDesc`` s.

    TRT calls the returned function during engine build to determine output
    shapes and dtypes.  The function:

    1. Detects whether the ``TensorDesc`` shape expressions are symbolic
       (placeholder / Fake) or concrete by inspecting ``shape_expr``.
    2. Converts each input ``TensorDesc`` to a ``torch.Tensor`` on the ``meta``
       device using the appropriate conversion helper.
    3. Calls ``meta_impl`` with the meta tensors to obtain output shape/dtype.
    4. Converts each output meta tensor back to a ``TensorDesc`` using the first
       input ``TensorDesc`` as a layout template.

    Args:
        meta_impl:   User-supplied meta function (same contract as ``custom_plugin``'s
                     ``meta_impl`` parameter).
        num_outputs: Expected number of outputs.
        num_inputs:  Number of input ``TensorDesc`` positional arguments.

    Returns:
        A callable suitable for use as the body of a ``@trtp.register`` function.

    Raises:
        TypeError: If ``meta_impl`` returns a non-tensor for any output position.
    """
    _meta_impl = meta_impl
    _num_outputs = num_outputs
    _num_inputs = num_inputs

    def _desc(*args: Any) -> Any:  # no type annotations — set via __signature__ below
        inp_args = args[:_num_inputs]
        shape_td0 = inp_args[0].shape_expr
        try:
            first_call = not shape_td0 or _is_symbolic_shape_expr(shape_td0)
        except (TypeError, ValueError):
            # shape_expr may not be iterable in some TRT versions; treat as
            # symbolic (first_call=True) so we use the symbolic path safely.
            first_call = True
        if logger.isEnabledFor(logging.DEBUG):
            for i, td in enumerate(inp_args):
                raw_shape = td.shape_expr
                raw_dtype = td.dtype
                logger.debug(
                    "raw TensorDesc input[%d] (before clamp): type=%s shape_expr=%s dtype=%s",
                    i, type(td).__name__, raw_shape, raw_dtype,
                )
            phase = (
                "TRT descriptor call with TensorDesc placeholder shape_expr (run meta_impl with symbolic inputs)"
                if first_call
                else "TRT descriptor call with concrete/symbolic shape_expr (run meta_impl)"
            )
            logger.debug("meta_impl %s", phase)
        if first_call:
            meta_tensors = [make_meta_tensor_from_td_symbolic(a) for a in inp_args]
        else:
            meta_tensors = [make_meta_tensor_from_td(a) for a in inp_args]
        if logger.isEnabledFor(logging.DEBUG):
            for i, (td, mt) in enumerate(zip(inp_args, meta_tensors)):
                logger.debug(
                    "meta_impl input[%d]: TensorDesc shape_expr=%s dtype=%s -> meta shape=%s dtype=%s",
                    i, list(td.shape_expr), td.dtype, tuple(mt.shape), mt.dtype,
                )
        meta_outs = _meta_impl(*meta_tensors)
        if not isinstance(meta_outs, (tuple, list)):
            meta_outs = (meta_outs,)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "meta_impl outputs: num_outputs=%d", len(meta_outs),
            )
        if _num_outputs == 1:
            out = meta_outs[0]
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    "meta_impl must return a torch.Tensor or sequence of tensors; "
                    f"got {type(out).__name__!r} for first output"
                )
            if first_call:
                return make_td_from_meta_using_template_symbolic(args[0], out)
            return make_td_from_meta_using_template(args[0], out)
        # Multi-output: build a TensorDesc for each output
        result = []
        for idx, out in enumerate(meta_outs[:_num_outputs]):
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    f"meta_impl output[{idx}] must be a torch.Tensor; "
                    f"got {type(out).__name__!r}"
                )
            if first_call:
                result.append(make_td_from_meta_using_template_symbolic(args[0], out))
            else:
                result.append(make_td_from_meta_using_template(args[0], out))
        return tuple(result)

    return _desc


def _build_desc_fn(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    num_outputs: int = 1,
) -> Callable[..., Any]:
    """Build the ``@trtp.register`` meta handler for a :class:`CustomPluginSpec`.

    Uses ``meta_impl`` (if provided) to infer output ``TensorDesc`` s; falls back
    to mirroring ``inp0`` (same shape/dtype) when ``meta_impl`` is ``None``.

    We use a ``*args`` closure and attach a custom ``inspect.Signature`` so that
    TRT's ``@trtp.register`` validation (``issubclass(param.annotation, TensorDesc)``)
    sees real ``TensorDesc`` class objects — not strings that would result from
    exec()-ing code in a module with ``from __future__ import annotations``.

    For ``num_outputs > 1``, the function returns a tuple of ``TensorDesc`` s (one
    per output), and ``meta_impl`` must return a tuple/list of that many tensors.

    Note: attrs are intentionally excluded from the descriptor signature.
    LIMITATION (workaround for NumPy 1.25+ incompatibility): TRT stores plugin
    field values as numpy arrays and calls ``attr_type_annot(f.data)`` to
    convert them back.  In NumPy 1.25+ this raises
    ``"only 0-dimensional arrays can be converted to Python scalars"`` for 1-D
    arrays.  Until TRT's plugin API is updated to handle NumPy 1.25+, we avoid
    the problem entirely by not registering any attrs.  Plugins that need to
    pass scalar hyperparameters (e.g. epsilon, axis) currently have no way to
    do so via the attrs mechanism and must bake constants into the kernel or
    pass them as additional tensor inputs.

    Args:
        descriptor:  The :class:`CustomPluginSpec` being registered.
        num_inputs:  Number of input ``TensorDesc`` positional parameters.
        num_outputs: Number of output ``TensorDesc`` s to produce (default 1).

    Returns:
        A callable with the correct ``inspect.Signature`` for ``@trtp.register``.
    """
    tensor_desc_cls = trtp.TensorDesc

    # meta_impl expects only the dynamic activation inputs, not the weight
    # inputs (which are TRT constant layers appended after activations).
    num_dynamic = num_inputs - len(descriptor.weights)
    _desc = _build_meta_impl_desc_fn(descriptor.meta_impl, num_outputs, num_dynamic)

    sig_params = _build_input_params(num_inputs, tensor_desc_cls)
    if num_outputs == 1:
        return_annotation = tensor_desc_cls
    else:
        # Build Tuple[TensorDesc, TensorDesc, ...] (n elements).
        # Subscripting typing.Tuple with a plain Python tuple of type args
        # unpacks them as separate type parameters (identical to writing
        # typing.Tuple[X, X] for n=2).  Do NOT use a list — that would
        # produce Tuple[List[X]] instead of Tuple[X, X].
        return_annotation = typing.Tuple[tuple([tensor_desc_cls] * num_outputs)]
    _desc.__signature__ = inspect.Signature(
        sig_params, return_annotation=return_annotation
    )
    return _desc


# ---------------------------------------------------------------------------
# Autotune function builder (format / tactic combinations via @trtp.autotune)
# ---------------------------------------------------------------------------

# Mapping from TRT dtype enum values to the string tokens accepted by
# AutoTuneCombination.  Populated lazily only when TRT is available.
_TRT_DTYPE_TOKEN: Dict[Any, str] = (
    {
        trt.float32: "FP32",
        trt.float16: "FP16",
        trt.bfloat16: "BF16",
        trt.int32: "INT32",
        trt.int8: "INT8",
    }
    if _TRT_AVAILABLE
    else {}
)

# Mapping from TRT TensorFormat enum values to AutoTuneCombination format tokens.
_TRT_FORMAT_TOKEN: Dict[Any, str] = (
    {
        trt.TensorFormat.LINEAR: "LINEAR",
        trt.TensorFormat.HWC8: "HWC8",
        trt.TensorFormat.HWC16: "HWC16",
        trt.TensorFormat.CHW4: "CHW4",
        trt.TensorFormat.CHW32: "CHW32",
        trt.TensorFormat.CHW16: "CHW16",
        trt.TensorFormat.DHWC8: "DHWC8",
    }
    if _TRT_AVAILABLE
    else {}
)


def _format_token_for_tactic(
    entry: TacticEntry,
    specs: List[KernelSpec],
) -> str:
    """Return the ``AutoTuneCombination`` format string for a single tactic entry.

    Looks up the first ``input_formats`` entry on the kernel spec selected by
    ``entry.spec_idx``.  Falls back to ``"LINEAR"`` if ``input_formats`` is
    absent or empty, or if the format is not in the known token map.

    Args:
        entry: The tactic table entry identifying the spec and config indices.
        specs: The full list of kernel specs from the descriptor.

    Returns:
        A format token string such as ``"LINEAR"`` or ``"CHW32"``.
    """
    spec = specs[entry.spec_idx]
    input_fmts = getattr(spec, "input_formats", None)
    if input_fmts:
        return _TRT_FORMAT_TOKEN.get(input_fmts[0], "LINEAR")
    return "LINEAR"


def _build_autotune_fn(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    num_outputs: int,
    tactic_table: List[TacticEntry],
) -> Optional[Callable[..., Any]]:
    """Build a ``@trtp.autotune`` function that registers dtype/format/tactic combinations.

    Returns ``None`` if the QDP autotune API (``trtp.autotune`` /
    ``trtp.AutoTuneCombination``) is not available in this TRT build.

    TRT calls this callback during engine build to enumerate valid
    (dtype, format, tactic) combinations.  It then benchmarks all valid
    combinations and passes the winning 1-based tactic ID to ``aot_impl``.

    ``AutoTuneCombination`` uses the string constructor::

        AutoTuneCombination(dtype_str, format_str, tactic_ids)

    where:

    * ``dtype_str`` — comma-separated per-I/O dtype options,
      e.g. ``"FP32|FP16, FP32|FP16"``
    * ``format_str`` — memory format, e.g. ``"LINEAR"``
    * ``tactic_ids`` — list of 1-based integer tactic IDs (0 is reserved by TRT)

    The autotune function signature must match::

        (inp0: TensorDesc, ..., inpN: TensorDesc,
         outputs: Tuple[TensorDesc]) -> List[AutoTuneCombination]

    Args:
        descriptor:    The :class:`CustomPluginSpec` being registered.
        num_inputs:    Number of input ``TensorDesc`` positional parameters.
        num_outputs:   Number of output tensors (used to iterate ``outputs``).
        tactic_table:  Pre-built list of :class:`TacticEntry` objects.

    Returns:
        A callable with the correct ``inspect.Signature`` for ``@trtp.autotune``,
        or ``None`` if the autotune API is unavailable.
    """
    if not (
        _TRT_AVAILABLE
        and hasattr(trtp, "autotune")
        and hasattr(trtp, "AutoTuneCombination")
    ):
        return None

    n_tactics = len(tactic_table)
    # Tactic IDs are 1-based: TRT reserves 0 as the "no-autotune" default.
    tactic_ids = list(range(1, n_tactics + 1))

    # Pre-compute format string per tactic so the closure doesn't need to
    # re-evaluate on every call to _autotune_fn.
    _tactic_formats = [
        _format_token_for_tactic(entry, descriptor.specs)
        for entry in tactic_table
    ]

    tensor_desc_cls = trtp.TensorDesc

    def _autotune_fn(*args: Any) -> List[Any]:
        inp_descs = list(args[:num_inputs])
        out_descs_raw = args[num_inputs]
        out_descs = list(out_descs_raw) if hasattr(out_descs_raw, "__iter__") else [out_descs_raw]
        parts = [
            _TRT_DTYPE_TOKEN.get(getattr(td, "dtype", None), "FP32")
            for td in inp_descs + out_descs
        ]
        dtype_str = ", ".join(parts)
        return [
            trtp.AutoTuneCombination(dtype_str, fmt, [tid])
            for tid, fmt in zip(tactic_ids, _tactic_formats)
        ]

    sig_params = (_build_input_params(num_inputs, tensor_desc_cls) +
                  [inspect.Parameter("outputs", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    _autotune_fn.__signature__ = inspect.Signature(sig_params)
    return _autotune_fn


# ---------------------------------------------------------------------------
# AOT impl builder (dispatches to backend)
# ---------------------------------------------------------------------------

# LIMITATION (TensorRT 10.14 bug — Blackwell only): Mixed Triton+CuTile or
# Triton+CuTeDSL plugins that produce SymIntExprs of different lengths crash
# with CUDA error 700 (illegal memory access) at execute_async_v3 on Blackwell
# (Myelin QUICKAOT path).  TRT picks a single SymIntExprs length for the whole
# plugin and passes that many extra int32s to every tactic's kernel; if the
# selected kernel has fewer .param declarations, the extras corrupt unrelated
# memory.  Workaround: do not mix Triton and CuTeDSL tactics in the same
# CustomPluginSpec on Blackwell.  Fix expected in a future TRT release.
def _build_aot_fn(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    tactic_table: List[TacticEntry],
) -> Callable[..., Any]:
    """Build a ``@trtp.aot_impl`` function that dispatches to the right backend.

    TRT calls this once with the winning tactic index (chosen by autotune).
    The first ``num_inputs`` positional args are ``TensorDesc`` inputs; then
    ``outputs`` (sequence of ``TensorDesc``); then ``tactic`` (int-like).

    The built function is cached in the thread-local ``_aot_fn_cache`` keyed by
    ``(op_name, num_inputs)`` to avoid redundant closure construction within a
    single thread.

    .. note:: **TensorRT 10.14 bug on Blackwell (Myelin QUICKAOT)**

        When tactics return ``SymIntExprs`` of different lengths, TRT/Myelin
        crashes with CUDA error 700 (illegal memory access) at
        ``execute_async_v3``.  Mixed Triton+CuTeDSL plugins trigger this because:

        - Triton: returns ``SymIntExprs(N)`` with N scalar kernel args (e.g.
          ``n_elements``)
        - CuTeDSL: returns ``SymIntExprs(0)`` — no scalar args, shape is in CuTe
          descriptors

        On Blackwell, TRT picks one ``SymIntExprs`` length for the entire plugin
        and passes that many extra ``int32`` s to every tactic's kernel at launch.
        If the selected kernel's PTX has fewer ``.param`` declarations than
        expected, the extras land in unrelated memory → crash.  On pre-Blackwell
        (non-Myelin), the mismatch is silently tolerated.

        No workaround is applied here; the annotation layer routes mixed-backend
        tests to pre-Blackwell GPUs (see ``tests/py/annotation/conftest.py``).
        Filed as a TensorRT bug; repro:
        ``tests/py/annotation/repro_blackwell_extra_len_mismatch.py``

    Args:
        descriptor:    The :class:`CustomPluginSpec` being registered.
        num_inputs:    Number of input ``TensorDesc`` positional arguments.
        tactic_table:  Pre-built list of :class:`TacticEntry` objects.

    Returns:
        A callable with the correct ``inspect.Signature`` for ``@trtp.aot_impl``.

    Raises:
        :class:`QDPRuntimeError`: If the tactic index is out of range or the
            kernel spec type is not supported.
    """
    cache = _get_aot_fn_cache()
    cache_key = (descriptor.op_name, num_inputs)
    if cache_key in cache:
        return cache[cache_key]

    op_name = descriptor.op_name
    tensor_desc_cls = trtp.TensorDesc

    def _aot_fn(*args: Any) -> Any:
        inp_descs = list(args[:num_inputs])
        if len(args) <= num_inputs:
            raise IndexError(
                f"AOT impl for {op_name!r} called with {len(args)} args but "
                f"expected at least {num_inputs + 1} "
                f"(inputs + outputs). TRT may not have passed the outputs "
                f"argument."
            )
        out_descs_raw = args[num_inputs]
        if hasattr(out_descs_raw, "__iter__"):
            out_descs = list(out_descs_raw)
        else:
            out_descs = [out_descs_raw]
        if len(args) > num_inputs + 1:
            tactic_id = int(args[num_inputs + 1])
        else:
            tactic_id = 1
            logger.debug(
                "aot_impl for %r: tactic_id not provided by TRT, defaulting to 1. "
                "This may indicate a TRT version mismatch.",
                op_name,
            )
        tactic_idx = tactic_id - 1

        if tactic_idx < 0 or tactic_idx >= len(tactic_table):
            raise QDPRuntimeError(
                op=op_name,
                stage="aot_impl",
                backend="custom_plugin",
                msg=(
                    f"tactic ID {tactic_id} (index {tactic_idx}) out of range "
                    f"[0, {len(tactic_table)}) for op {op_name!r}"
                ),
            )
        entry = tactic_table[tactic_idx]
        spec = descriptor.specs[entry.spec_idx]
        configs = spec.configs if spec.configs else [{}]
        cfg = configs[entry.config_idx]
        launch_fn = spec.launch_fn

        sym_inputs = [
            SymbolicTensor(td=td, role=TensorRole.INPUT, index=i)
            for i, td in enumerate(inp_descs)
        ]
        sym_outputs = [
            SymbolicTensor(td=td, role=TensorRole.OUTPUT, index=j)
            for j, td in enumerate(out_descs)
        ]
        host_args = sym_inputs + sym_outputs

        plugin_attrs = descriptor.attrs

        if isinstance(spec, TritonSpec):
            from ._aot._triton import aot_impl_triton
            return aot_impl_triton(
                qdp_symbol=op_name, spec=spec, cfg=cfg, launch_fn=launch_fn,
                host_args=host_args, inp_descs=inp_descs, out_descs=out_descs,
                attrs=plugin_attrs,
            )
        elif isinstance(spec, CuTileSpec):
            from ._aot._cutile import aot_impl_cutile
            return aot_impl_cutile(
                qdp_symbol=op_name, spec=spec, cfg=cfg, launch_fn=launch_fn,
                host_args=host_args, inp_descs=inp_descs, out_descs=out_descs,
                attrs=plugin_attrs,
            )
        elif isinstance(spec, CuTeDSLSpec):
            from ._aot._cutedsl import aot_impl_cutedsl
            return aot_impl_cutedsl(
                qdp_symbol=op_name, spec=spec, cfg=cfg, launch_fn=launch_fn,
                host_args=host_args, inp_descs=inp_descs, out_descs=out_descs,
                attrs=plugin_attrs,
            )
        else:
            raise QDPRuntimeError(
                op=op_name, stage="aot_impl", backend="custom_plugin",
                msg=f"unsupported kernel spec type {type(spec).__name__!r} for op {op_name!r}",
            )

    sig_params = (_build_input_params(num_inputs, tensor_desc_cls) +
                  [inspect.Parameter("outputs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                   inspect.Parameter("tactic",  inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    _aot_fn.__signature__ = inspect.Signature(
        sig_params,
        return_annotation=typing.Tuple[
            typing.Union[str, bytes],
            typing.Union[str, bytes],
            trtp.KernelLaunchParams,
            trtp.SymIntExprs,
        ],
    )

    cache[cache_key] = _aot_fn
    return _aot_fn


# ---------------------------------------------------------------------------
# QDP registration
# ---------------------------------------------------------------------------


def register_custom_plugin(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    num_outputs: int = 1,
    qdp_name: Optional[str] = None,
) -> None:
    """Register a :class:`CustomPluginSpec` with TRT's QDP plugin registry.

    Registers three QDP callbacks under ``qdp_name`` (if provided) or
    ``descriptor.op_name``:

    * ``@trtp.register``  — shape/dtype descriptor (uses ``meta_impl`` or identity)
    * ``@trtp.autotune``  — format/tactic combinations per I/O position
    * ``@trtp.aot_impl``  — AOT kernel dispatch (Triton/CuTile/CuTeDSL backend)

    **Idempotency**: repeated calls for the same ``op_name`` are no-ops.  The
    function uses a double-checked locking pattern against
    ``_qdp_registered_ops`` (a process-global set) to guard against concurrent
    registration from multiple threads.  See the module-level comment on
    ``_qdp_registered_ops`` for the full threading contract.

    If TRT's own registry reports that an op is "already registered" (e.g.
    because a prior in-process call registered it before ``_qdp_registered_ops``
    was updated), the function catches the error, logs a debug message, and
    marks the op as registered so future calls are no-ops.

    Args:
        descriptor:  The :class:`CustomPluginSpec` to register.
        num_inputs:  Number of TRT input tensors for this op. MUST include weight
            tensors (from trt.add_constant) in addition to activation inputs. The
            function computes num_dynamic = num_inputs - len(descriptor.weights)
            internally to pass only activation count to meta_impl.
        num_outputs: Number of TRT output tensors (default 1).
        qdp_name:    Optional override for the QDP registration name.  When
                     provided the plugin is registered under this name instead
                     of ``descriptor.op_name``.  Use this when wiring a
                     :class:`CustomPluginSpec` to a ``torch.library`` op whose
                     name differs from the TTA fingerprint name (e.g. when
                     called from ``trt_plugins.custom_op``).

    Raises:
        Exception: Any exception raised by TRT's ``trtp.register`` that is
            **not** an "already registered" error is re-raised verbatim.
        Exception: Any exception raised by TRT's ``trtp.aot_impl`` that is
            **not** an "already registered" error is re-raised verbatim.
    """
    op_name = qdp_name if qdp_name is not None else descriptor.op_name

    # Fast path: check without the lock first (common case after first registration).
    # Safe because _qdp_registered_ops only grows — see module-level threading contract.
    if op_name in _qdp_registered_ops:
        return

    with _qdp_registration_lock:
        # Re-check inside the lock to handle the race between the fast-path
        # check above and acquiring the lock (TOCTOU).
        if op_name in _qdp_registered_ops:
            return

        tactic_table = build_tactic_table(descriptor.specs)

        # 1. Register shape/dtype descriptor.
        #    TRT's internal QDP registry is process-global; if TRT already knows
        #    the op (from a prior call in this process), catch "already has a
        #    definition" and mark it registered without re-registering.
        desc_fn = _build_desc_fn(descriptor, num_inputs, num_outputs)
        try:
            trtp.register(op_name)(desc_fn)
        except Exception as exc:  # noqa: BLE001
            # TRT does not expose a typed "already registered" exception; the
            # only signal is a substring of the error message.  Any other
            # exception is a genuine registration failure and must propagate.
            if "already" in str(exc).lower():
                logger.debug(
                    "QDP op %r already registered in TRT registry, skipping: %s",
                    op_name, exc,
                )
                _qdp_registered_ops.add(op_name)
                return
            raise

        # 2. Register autotune handler (format/tactic combinations).
        #    _build_autotune_fn returns None if QDP autotune API is unavailable.
        autotune_fn = _build_autotune_fn(descriptor, num_inputs, num_outputs, tactic_table)
        if autotune_fn is not None:
            try:
                trtp.autotune(op_name)(autotune_fn)
            except Exception as exc:  # noqa: BLE001
                # Autotune registration failures are non-fatal: TRT will fall back
                # to a single default tactic.  Log at debug and continue.
                if "already" in str(exc).lower():
                    logger.debug(
                        "QDP autotune %r already registered in TRT registry; skipping",
                        op_name,
                    )
                else:
                    logger.debug(
                        "QDP autotune registration failed for %r (non-fatal, "
                        "TRT will use default tactic): %s",
                        op_name, exc,
                    )

        # 3. Register AOT implementation (real kernel dispatch, no Python fallback).
        aot_fn = _build_aot_fn(descriptor, num_inputs, tactic_table)
        try:
            trtp.aot_impl(op_name)(aot_fn)
        except Exception as exc:  # noqa: BLE001
            # Same "already registered" pattern as step 1.  Any other exception
            # is a genuine failure (e.g. PTX compilation error) and must propagate.
            if "already" in str(exc).lower():
                logger.debug(
                    "QDP aot_impl %r already registered in TRT registry; skipping",
                    op_name,
                )
            else:
                raise

        _qdp_registered_ops.add(op_name)
        logger.debug(
            "Registered QDP plugin %r (num_inputs=%d, num_outputs=%d, tactics=%d)",
            op_name,
            num_inputs,
            num_outputs,
            len(tactic_table),
        )


# ---------------------------------------------------------------------------
# TRT network lowering
# ---------------------------------------------------------------------------


def lower_custom_plugin_descriptor(
    ctx: Any,
    descriptor: CustomPluginSpec,
    trt_inputs: List[Any],
    name: str,
) -> Union[Any, Sequence[Any]]:
    """Lower a :class:`CustomPluginSpec` to a TRT ``IPluginV3`` layer.

    This is the final step in the :class:`CustomPluginSpec` lifecycle (see
    class docstring).  It:

    1. Injects weight tensors as ``trt.add_constant`` layers, appending them to
       ``trt_inputs`` so the plugin sees ``(*activations, *weights)`` as inputs.
    2. Calls :func:`register_custom_plugin` (idempotent) to ensure the QDP
       callbacks are registered before the network refers to the op.
    3. Resolves ``trtp.op.<namespace>.<plugin_name>`` and calls it with
       ``trt_inputs`` to obtain the plugin handle.
    4. Adds the plugin layer via ``ctx.net.add_plugin(..., aot=True)`` and
       attaches TTA layer metadata for debugging.
    5. Returns a single ``trt.ITensor`` (for single-output plugins) or a tuple
       of ``trt.ITensor`` s.

    Args:
        ctx:        Torch-TRT ``ConversionContext`` providing ``ctx.net``.
        descriptor: :class:`CustomPluginSpec` for this op.
        trt_inputs: List of ``trt.ITensor`` dynamic activation inputs.
        name:       Layer name used for TRT network debugging.

    Returns:
        A single ``trt.ITensor`` if the plugin has one output, or a ``tuple``
        of ``trt.ITensor`` s for multi-output plugins.

    Raises:
        :class:`QDPRuntimeError`: Wraps any non-QDP exception raised during
            lowering (registration, plugin construction, or layer addition).
    """
    op_name = descriptor.op_name

    try:
        # num_outputs was pre-computed at custom_plugin() creation time.
        num_outputs = descriptor.num_outputs

        # Weight binding: tensor-valued kwargs declared in custom_plugin() are injected
        # here as TRT constant layers, appended after the dynamic activation inputs.
        # The annotated function only receives the activations (no weight args), so the
        # eager body is unchanged.  The launch_fn contract is:
        #   (*activations, *weights_in_declaration_order, *outputs, ...)
        if descriptor.weights:
            import numpy as np
            weight_trt_tensors = []
            for wname, wtensor in descriptor.weights.items():
                np_arr = wtensor.detach().cpu().contiguous().numpy()
                trt_weights = trt.Weights(np_arr)
                const_layer = ctx.net.add_constant(tuple(np_arr.shape), trt_weights)
                const_layer.name = f"{name}_weight_{wname}"
                weight_trt_tensors.append(const_layer.get_output(0))
            trt_inputs = list(trt_inputs) + weight_trt_tensors

        num_inputs = len(trt_inputs)
        register_custom_plugin(descriptor, num_inputs, num_outputs)

        # Parse namespace and plugin name from op_name (format: "ns::op").
        if "::" in op_name:
            namespace, plugin_name = op_name.split("::", 1)
        else:
            namespace = "tta_custom"
            plugin_name = op_name

        ns_module = getattr(trtp.op, namespace)
        plugin_fn = getattr(ns_module, plugin_name)

        # Attrs are baked into the kernel PTX at AOT time; do not pass as TRT
        # plugin fields (TRT stores them as numpy arrays and the round-trip
        # float(np.array([v])) fails in NumPy 1.25+).
        plugin_layer = ctx.net.add_plugin(plugin_fn(*trt_inputs), aot=True)
        plugin_layer.name = name

        # Sanity check: TRT plugin layer output count must match the descriptor's
        # pre-computed num_outputs.  A mismatch here means meta_impl inferred the
        # wrong count (e.g., because a dispatch mode was active at creation time).
        if plugin_layer.num_outputs != descriptor.num_outputs:
            raise QDPRuntimeError(
                op=op_name,
                stage="lowering",
                backend="custom_plugin",
                msg=(
                    f"TRT plugin layer has {plugin_layer.num_outputs} outputs but descriptor "
                    f"expected {descriptor.num_outputs}. This likely means meta_impl inferred "
                    f"num_outputs incorrectly (e.g., while a dispatch mode was active). "
                    f"Pass num_outputs= explicitly to custom_plugin()."
                ),
            )

        _first_spec = descriptor.specs[0] if descriptor.specs else None
        if isinstance(_first_spec, TritonSpec):
            _backend = "triton"
        elif isinstance(_first_spec, CuTileSpec):
            _backend = "cutile"
        elif isinstance(_first_spec, CuTeDSLSpec):
            _backend = "cutedsl"
        else:
            _backend = "custom_plugin"
        _fn_specs = []
        for _s in descriptor.specs:
            _fn = getattr(getattr(_s, "launch_fn", None), "__name__", None)
            if _fn:
                _cfgs = _s.configs if getattr(_s, "configs", None) else [{}]
                for _cfg in _cfgs:
                    _fn_specs.append((_fn, _cfg))
        set_tta_layer_metadata(plugin_layer, _backend, plugin_name, name,
                               attrs=descriptor.attrs or None,
                               fn_specs=_fn_specs or None)

        if plugin_layer.num_outputs == 1:
            return plugin_layer.get_output(0)
        return tuple(
            plugin_layer.get_output(i) for i in range(plugin_layer.num_outputs)
        )

    except Exception as exc:
        if isinstance(exc, QDPRuntimeError):
            raise
        raise QDPRuntimeError(
            op=op_name,
            stage="compile",
            backend="custom_plugin",
            msg=f"lowering failed for op {op_name!r} (layer {name!r}): {exc}",
        ) from exc
