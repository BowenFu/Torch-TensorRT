"""
Torch-TensorRT Annotation Layer (TTA) — public API module.

The annotation layer provides a declarative API for embedding TensorRT-specific
lowering hints directly into PyTorch model code.  Annotations are eager no-ops
(the model runs normally under ``torch.eager`` and ``torch.compile``) and are
only activated during ``tta.compile()``, which drives the following pipeline:

1. **Annotate** — model methods are decorated with ``tta.export_as`` or
   ``tta.lower_as``.
2. **Export** — ``tta.compile()`` calls ``torch.export.export`` with graph-
   tagging hooks active.  Boundary ops (``export_as``) and region records
   (``lower_as``) are stamped into the exported ``ExportedProgram``.
3. **lower_as pass** — ``tta.lower_as`` regions are rewritten so that the
   affected subgraph is replaced by a TRT-native op.
4. **TRT compilation** — ``torch_tensorrt.dynamo.compile`` builds the TRT
   engine from the (possibly rewritten) exported program.

Primary entry points
--------------------
- ``tta.export_as``   — annotate a function/method with a TRT implementation.
- ``tta.lower_as``    — replace a region body with a TRT implementation.
- ``tta.compile``     — export, annotate, and compile to a TRT engine.

Usage::

    import torch_tensorrt.annotation as tta

    # Plugin annotation
    @tta.export_as(impl=tta.plugin("MyPlugin", "1.0", "my_namespace"))
    def my_custom_op(x, y):
        return x + y

    # Builtin annotation
    @tta.export_as(impl=tta.builtin("convolution_nd", num_output_maps=64))
    def my_conv(x):
        return x

    # Custom Triton kernel
    @tta.export_as(impl=tta.custom_plugin(kernel=my_triton_kernel, out_dtypes=[torch.float32]))
    def my_kernel_op(x):
        return my_triton_kernel(x)
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

import functools
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import numpy as np
import torch

# ---------------------------------------------------------------------------
# TTA internal imports — error types
# ---------------------------------------------------------------------------

# Structured error types
from ._errors import TTADiagnosticError, TTABuiltinError, TTAPluginError, LowerAsError

# ---------------------------------------------------------------------------
# TTA internal imports — spec types and factory functions
# ---------------------------------------------------------------------------

from ._specs import (
    AnnotationMetadata,
    BuiltinSpec,
    KernelImplSpec,
    CuTeDSLSpec,
    CuTileSpec,
    FromMethodSelf,  # internal — not re-exported
    RegistryPluginSpec,
    TritonSpec,
    attach_annotation_metadata,
    builtin,
    cutedsl,
    cutile,
    self_attr,
    get_annotation_metadata,
    normalize_impl_to_spec,
    plugin,
    triton,
)

# ---------------------------------------------------------------------------
# TTA internal imports — custom plugin descriptor
# ---------------------------------------------------------------------------

# custom_plugin now lives in _custom_plugin._descriptor and returns CustomPluginSpec
from ._custom_plugin._descriptor import CustomPluginSpec, custom_plugin

# ---------------------------------------------------------------------------
# TTA internal imports — lower_as
# ---------------------------------------------------------------------------

from ._lower_as_api import clear_lower_as_regions, get_all_lower_as_regions, lower_as, spec_to_impl_id
from . import _lower_as_pass  # noqa: F401
from torch_tensorrt.dynamo._compiler import (
    register_compile_pass as _register_compile_pass,
    register_preserved_ep_attr as _register_preserved_ep_attr,
)

# Preserve ep._tta across run_decompositions() so that compile passes can
# read annotation IR metadata.
_register_preserved_ep_attr("_tta")

# Pre-tagging pass: must run BEFORE lower_as rewrites so that
# trt_layer_metadata is set on the correct nodes.
from . import _pre_tagging_pass as _pre_tagging_pass_mod  # noqa: F401
_register_compile_pass(_pre_tagging_pass_mod.apply_pre_tagging)

_register_compile_pass(_lower_as_pass.apply_lower_as_regions)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Diagnostic error types
    "TTADiagnosticError",
    "TTABuiltinError",
    "TTAPluginError",
    "LowerAsError",
    # Implementation descriptor types
    "AnnotationMetadata",
    "BuiltinSpec",
    "RegistryPluginSpec",
    "TritonSpec",
    "CuTileSpec",
    "CuTeDSLSpec",
    "CustomPluginSpec",
    # Annotation decorators
    "export_as",
    "lower_as",
    # Implementation factory functions
    "plugin",
    "builtin",
    "self_attr",
    "custom_plugin",
    "triton",
    "cutile",
    "cutedsl",
]


# ---------------------------------------------------------------------------
# Internal helpers — spec resolution
# ---------------------------------------------------------------------------


def _to_numpy_f32(val: Any) -> Any:
    """Convert a ``torch.Tensor`` to a ``numpy.ndarray`` with dtype ``float32``.

    Non-tensor values are returned unchanged.  Used when resolving
    ``tta.self_attr(...)`` parameters that hold weight tensors.

    Args:
        val: Any value.  If it is a ``torch.Tensor``, it is detached, moved to
            CPU, and cast to ``float32`` before being returned as a numpy array.

    Returns:
        ``numpy.ndarray`` (float32) if *val* is a tensor, otherwise *val* as-is.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().astype(np.float32)
    return val


def _resolve_from_method_self(
    spec: Union[RegistryPluginSpec, BuiltinSpec, Any],
    self_obj: object,
) -> Union[RegistryPluginSpec, BuiltinSpec, Any]:
    """Resolve ``FromMethodSelf`` markers in *spec* using a live module instance.

    ``tta.self_attr("attr.path")`` is used as a parameter default to declare
    that a module attribute (e.g., ``self.conv.weight``) should be appended to
    the boundary op inputs at tracing time.  Before tracing, these markers must
    be replaced with their actual tensor values (converted to ``float32`` numpy
    arrays for TRT attribute passing).

    For ``RegistryPluginSpec``: resolves ``attrs``.
    For ``BuiltinSpec``: resolves ``kwargs``.

    Args:
        spec: A ``RegistryPluginSpec`` or ``BuiltinSpec`` that may contain
            ``FromMethodSelf`` marker values.  Other spec types are returned
            unchanged.
        self_obj: A live ``nn.Module`` (or any object) whose attributes are
            read to fill in the markers.

    Returns:
        A new spec of the same type with all ``FromMethodSelf`` values replaced
        by the resolved (and converted) attribute values, or the original *spec*
        if no markers were present.

    Raises:
        AttributeError: If *self_obj* does not have an attribute named by a
            ``FromMethodSelf`` marker.
        RuntimeError: If tensor-to-numpy conversion fails for a weight tensor.
    """
    if isinstance(spec, RegistryPluginSpec) and any(
        isinstance(v, FromMethodSelf) for v in spec.attrs.values()
    ):
        resolved_attrs: Dict[str, Any] = {}
        for k, v in spec.attrs.items():
            if isinstance(v, FromMethodSelf):
                resolved_attrs[k] = _to_numpy_f32(getattr(self_obj, v.attr_name))
            else:
                resolved_attrs[k] = v
        return RegistryPluginSpec(
            name=spec.name,
            version=spec.version,
            namespace=spec.namespace,
            attrs=resolved_attrs,
        )
    elif isinstance(spec, BuiltinSpec) and any(
        isinstance(v, FromMethodSelf) for v in spec.kwargs.values()
    ):
        resolved_kwargs: Dict[str, Any] = {}
        for k, v in spec.kwargs.items():
            if isinstance(v, FromMethodSelf):
                resolved_kwargs[k] = _to_numpy_f32(getattr(self_obj, v.attr_name))
            else:
                resolved_kwargs[k] = v
        return BuiltinSpec(
            add_name=spec.add_name,
            kwargs=resolved_kwargs,
        )
    else:
        return spec


def _spec_has_from_method_self(spec: Any) -> bool:
    """Return ``True`` if *spec* contains any unresolved ``FromMethodSelf`` markers.

    Used as a guard to detect when ``tta.self_attr(...)`` parameters have not
    yet been resolved from a live module instance before tracing begins.

    Args:
        spec: Any impl spec.  ``RegistryPluginSpec`` and ``BuiltinSpec`` are inspected;
            all other types return ``False``.

    Returns:
        ``True`` if at least one value in the spec's attrs/kwargs is a
        ``FromMethodSelf`` instance, ``False`` otherwise.
    """
    if isinstance(spec, RegistryPluginSpec):
        return any(isinstance(v, FromMethodSelf) for v in spec.attrs.values())
    if isinstance(spec, BuiltinSpec):
        return any(isinstance(v, FromMethodSelf) for v in spec.kwargs.values())
    return False


def _get_nested_attr(obj: object, path: str) -> Any:
    """Traverse a dotted attribute path on *obj* and return the leaf value.

    Example: ``_get_nested_attr(module, "conv.weight")`` is equivalent to
    ``module.conv.weight``.

    Args:
        obj: Root object to traverse.
        path: Dot-separated attribute path (e.g., ``"encoder.proj.weight"``).

    Returns:
        The attribute value at the end of the path.

    Raises:
        AttributeError: If any segment of the path is not found on the
            intermediate object.
    """
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _collect_self_attr_params(fn: Callable) -> List[FromMethodSelf]:
    """Return ordered list of ``FromMethodSelf`` defaults from *fn*'s signature.

    Parameters with a ``tta.self_attr(...)`` default declare module-level tensors
    that should be appended to the boundary op inputs during tracing.  The
    function signature serves as the contract::

        def conv_relu_pool(self, x,
                           weight=tta.self_attr("conv.weight"),
                           bias=tta.self_attr("conv.bias")):
            out = self.conv(x)   # module's conv2d
            ...

    Args:
        fn: Any callable.  Parameters without a ``FromMethodSelf`` default are
            silently skipped.  If the signature cannot be introspected (e.g.,
            built-in C functions), an empty list is returned.

    Returns:
        Ordered list of ``FromMethodSelf`` instances, one per matching parameter,
        in declaration order.
    """
    import inspect as _inspect
    result: List[FromMethodSelf] = []
    try:
        for param in _inspect.signature(fn).parameters.values():
            if isinstance(param.default, FromMethodSelf):
                result.append(param.default)
    except (ValueError, TypeError):
        # ValueError: signature not introspectable (e.g., built-in).
        # TypeError:  non-callable passed (defensive; should not happen in normal usage).
        pass
    return result


def _spec_impl_id(spec: Any) -> str:
    """Return a human-readable impl identity string for diagnostics.

    The string is used in error/warning messages to identify the impl that
    caused a problem.  Format varies by spec type:

    - ``RegistryPluginSpec`` → ``"<name> v<version> (<namespace>)"``
    - ``BuiltinSpec`` → ``"<add_name>"``
    - Other specs with an ``op_name`` attr → ``"<op_name>"``
    - Fallback → ``repr(type(spec).__name__)``

    Args:
        spec: Any impl spec or descriptor object.

    Returns:
        Human-readable string identifying the impl.
    """
    if isinstance(spec, RegistryPluginSpec):
        return f"{spec.name} v{spec.version} ({spec.namespace})"
    if isinstance(spec, BuiltinSpec):
        return spec.add_name
    # CustomPluginSpec / KernelImplSpec / etc.
    if hasattr(spec, "op_name"):
        return spec.op_name
    return repr(type(spec).__name__)


# ---------------------------------------------------------------------------
# export_as decorator
# ---------------------------------------------------------------------------


def export_as(
    impl: Union[RegistryPluginSpec, BuiltinSpec, "KernelImplSpec", Any],
    require: bool = False,
    name: Optional[str] = None,
    **ctor_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """Annotate a callable with a TensorRT implementation descriptor.

    The decorated function behaves identically in eager mode.  During
    ``torch.export`` (driven by ``tta.compile``), callsites are intercepted
    and replaced with boundary leaf ops that lower to TensorRT layers.

    Args:
        impl: Implementation descriptor — one of:

            - ``tta.plugin(name, version, namespace, **attrs)``
            - ``tta.builtin(add_name, **kwargs)``
            - ``tta.custom_plugin(tta.triton(...))`` / ``tta.cutile(...)`` /
              ``tta.cutedsl(...)``

        require: If ``True``, compilation *must* successfully lower this
            boundary op to the specified implementation.  A failure (plugin
            not registered, builtin creation error, IO mismatch) raises
            immediately with an error identifying the op name, impl identity,
            and reason.  If ``False`` (default), failures are logged as
            warnings and the op falls back to the PyTorch implementation.

        name: Optional human-readable label used in diagnostics, TRT layer
            naming, and attribution.  Defaults to ``None``.

        **ctor_kwargs: Additional decorator options (reserved for future use).

    **Instance methods and ``nn.Module`` weights**

    When the decorated function is an instance method (has ``self``), use
    ``tta.self_attr("attr.path")`` as parameter defaults to declare which
    module tensors should flow into the plugin as extra inputs.  The body can
    then use ``self.conv(x)`` (the module's ``nn.Conv2d``) directly::

        @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_fn, configs=[...])))
        def conv_relu_pool(self, x,
                           weight=tta.self_attr("conv.weight"),
                           bias=tta.self_attr("conv.bias")):
            out = self.conv(x)   # module's conv2d, not F.conv2d
            return F.max_pool2d(F.relu(out), 2, 2)

        def forward(self, x):
            return self.conv_relu_pool(x)   # only x — weights come from self_attr

    During tracing the ``FromMethodSelf`` defaults are resolved from ``self``
    and appended to the boundary op args in declaration order.

    Returns:
        A decorator that attaches ``AnnotationMetadata`` to the callable and
        wraps it so that tracing-time calls are routed through the boundary op.

    Raises:
        TTADiagnosticError: During tracing, if ``FromMethodSelf`` attrs were
            not resolved before export, or if boundary op registration fails
            unexpectedly.

    Examples:
        >>> @export_as(impl=plugin("MyPlugin", "1.0", "my_ns", alpha=0.5))
        ... def my_op(x):
        ...     return x * 2

        >>> @export_as(impl=builtin("add_activation", type=trt.ActivationType.RELU))
        ... def my_relu(x):
        ...     return torch.relu(x)
    """
    from ._registry import get_or_create_op_for_boundary, set_require_for_op

    spec = normalize_impl_to_spec(impl)
    metadata = AnnotationMetadata(impl=spec, kwargs=ctor_kwargs, name=name, require=require)
    needs_resolve: bool = _spec_has_from_method_self(spec)

    def decorator(fn: Callable) -> Callable:
        attach_annotation_metadata(fn, metadata)
        # Cache slot for the once-resolved spec (populated on first eager forward pass).
        resolved_spec_cache: List[Optional[Any]] = [None]
        # Detect self_attr(...) defaults — these declare module tensor inputs
        # for instance methods that use self.conv(x) in the body.
        _self_attr_inputs: List[FromMethodSelf] = _collect_self_attr_params(fn)
        # Also collect self_attr(...) kwargs declared in CustomPluginSpec
        # (e.g. tta.custom_plugin(..., w=tta.self_attr("scale"))).
        if hasattr(spec, "self_attrs") and spec.self_attrs:
            _self_attr_inputs = _self_attr_inputs + list(spec.self_attrs.values())

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # --- Eager spec resolution (runs once on first non-tracing forward) ---
            if needs_resolve and resolved_spec_cache[0] is None and len(args) > 0:
                first_arg = args[0]
                if not isinstance(first_arg, torch.fx.Proxy):
                    resolved_spec_cache[0] = _resolve_from_method_self(spec, first_arg)

            # --- Tracing path: replace with boundary op call ---
            if torch._dynamo.is_compiling() or any(
                isinstance(arg, torch.fx.Proxy)
                or (
                    isinstance(arg, torch.Tensor)
                    and type(arg).__module__.startswith("torch._subclasses")
                )
                for arg in args
            ):
                # Collect explicit tensor args from the call site (skip non-tensor self).
                tensor_args: List[Any] = [
                    arg for arg in args if isinstance(arg, (torch.Tensor, torch.fx.Proxy))
                ]
                # Append self_attr(...) tensor inputs declared as parameter defaults.
                if _self_attr_inputs:
                    self_obj = next(
                        (a for a in args if not isinstance(a, (torch.Tensor, torch.fx.Proxy))),
                        None,
                    )
                    if self_obj is not None:
                        for fms in _self_attr_inputs:
                            tensor_args.append(_get_nested_attr(self_obj, fms.attr_name))
                runtime_input_arity: int = len(tensor_args)
                spec_to_use = resolved_spec_cache[0] if resolved_spec_cache[0] is not None else spec
                if _spec_has_from_method_self(spec_to_use):
                    raise TTADiagnosticError(
                        "tta.export_as: self_attr attrs were not resolved before tracing. "
                        "Ensure the model runs at least one eager forward pass before "
                        "torch.export so that tta.self_attr(...) parameters can be "
                        "read from the live module.",
                        stage="export",
                        leaf_op=getattr(fn, "__qualname__", getattr(fn, "__name__", None)),
                        impl_id=_spec_impl_id(spec_to_use),
                    )
                try:
                    boundary_op = get_or_create_op_for_boundary(spec_to_use, runtime_input_arity)
                except TTADiagnosticError:
                    raise
                except (RuntimeError, ValueError, TypeError) as _e:
                    # Re-raise with full context: op name, impl identity, arity.
                    raise TTADiagnosticError(
                        f"tta.export_as: failed to register boundary op during export "
                        f"(input_arity={runtime_input_arity}): {_e}",
                        stage="export",
                        leaf_op=getattr(fn, "__qualname__", getattr(fn, "__name__", None)),
                        impl_id=_spec_impl_id(spec_to_use),
                    ) from _e
                op_overload = getattr(boundary_op, "default", boundary_op)
                set_require_for_op(op_overload, require)
                # Record op → name in capture state so _stamp_export_as_boundary_nodes
                # can set layer_metadata after export (result is a FakeTensor during
                # torch.export.export, not a Proxy, so result.node is not accessible).
                if name:
                    from ._capture_state import (
                        in_capture_mode as _in_capture_mode,
                        record_export_as_op_name as _record_export_as_op_name,
                    )
                    if _in_capture_mode():
                        _record_export_as_op_name(str(op_overload), name)
                result = boundary_op(*tensor_args)
                # Fallback: if result happens to be a Proxy (e.g. symbolic_trace),
                # also stamp directly on the node for that tracing path.
                if name and hasattr(result, "node"):
                    result.node.meta["layer_metadata"] = name
                return result

            # --- Eager path: run the original function unchanged ---
            return fn(*args, **kwargs)

        return wrapper

    return decorator

# ---------------------------------------------------------------------------
# TTA integration — hooks into torch_tensorrt.compile transparently
# ---------------------------------------------------------------------------

from contextlib import contextmanager as _contextmanager


@_contextmanager
def _tta_export_context(model: Any, inputs: Any) -> Any:
    """Activate TTA capture mode around torch.export.export."""
    from ._capture_state import CAPTURE_MODE, CaptureMode
    from ._capture_state import (
        in_capture_mode,
        install_graph_tagging,
        reset_region_table,
        set_capture_mode,
        uninstall_graph_tagging,
    )
    if in_capture_mode():
        yield
        return
    reset_region_table()
    clear_lower_as_regions()
    _cm_token = CAPTURE_MODE.set(CaptureMode(kind="amp"))
    set_capture_mode(True)
    install_graph_tagging()
    try:
        yield
    finally:
        uninstall_graph_tagging()
        set_capture_mode(False)
        CAPTURE_MODE.reset(_cm_token)


def _tta_post_trace_hook(exported_program: Any, inputs: Any) -> None:
    """Build TTA annotation IR after torch.export; registered as a post-trace hook."""
    import torch._ops as _torch_ops
    from ._capture_state import (
        get_export_as_op_names,
        get_impl_registry,
        get_region_table,
        reset_region_table,
    )
    from ._compile.pipeline import build_annotation_ir
    from ._lower_as_api import clear_lower_as_regions, get_all_lower_as_regions
    from ._registry import get_require_for_op  # noqa: F401

    # Stamp export_as boundary op nodes with name → layer_metadata.
    _has_boundary_ops = any(
        node.op == "call_function"
        and isinstance(node.target, _torch_ops.OpOverload)
        and node.target.namespace.startswith("torch_tensorrt_anno_")
        for node in exported_program.graph.nodes
    )
    if _has_boundary_ops:
        _op_to_name = get_export_as_op_names()
        _op_name_map: Dict[str, str] = {}
        for node in exported_program.graph.nodes:
            if (
                node.op == "call_function"
                and isinstance(node.target, _torch_ops.OpOverload)
                and node.target.namespace.startswith("torch_tensorrt_anno_")
            ):
                _nname = _op_to_name.get(str(node.target))
                if _nname:
                    node.meta["layer_metadata"] = _nname
                    _op_name_map[str(node.target)] = _nname
        if _op_name_map:
            _tta = getattr(exported_program, "_tta", {}) or {}
            _merged = dict(_tta.get("lower_as_op_to_name") or {})
            _merged.update(_op_name_map)
            _tta["lower_as_op_to_name"] = _merged
            exported_program._tta = _tta

    # Populate ep._tta with lower_as region table and region views.
    _lower_as_table: Dict[int, Any] = {}
    _impl_registry_la: Dict[str, Any] = {}
    for region_id, cfg in get_all_lower_as_regions().items():
        _impl_id = spec_to_impl_id(cfg.impl)
        _lower_as_table[region_id] = {
            "id": region_id,
            "kind": "lower_as",
            "name": cfg.name or f"lower_as_region_{region_id}",
            "mode": None,
            "constraints": {"require": cfg.require},
            "args": {"impl_id": _impl_id},
        }
        _impl_registry_la[_impl_id] = cfg.impl

    if _lower_as_table:
        _gm = exported_program.graph_module
        _node_regions_la: Dict[Any, Any] = {}
        for _n in _gm.graph.nodes:
            _stack = _n.meta.get("tta_regions")
            if _stack:
                _node_regions_la[_n] = list(_stack)
        from .ir.region_view import build_region_views_post_export as _build_rvs
        _la_rvs = _build_rvs(
            _gm,
            node_regions=_node_regions_la,
            region_ids=list(_lower_as_table.keys()),
        )
        _tta = getattr(exported_program, "_tta", {}) or {}
        _tta["region_table"] = _lower_as_table
        _tta["impl_registry"] = _impl_registry_la
        _tta["region_views"] = _la_rvs
        exported_program._tta = _tta

        # Apply the lower_as rewrite NOW, while node references in RegionView
        # are still valid.  run_decompositions() (called inside dynamo_compile)
        # returns a fresh ExportedProgram whose FX nodes differ from the ones
        # stored in RegionView, so the compile-time pass would find no live
        # nodes and silently skip.  Rewriting here ensures the TTA leaf op is
        # in the graph before decompositions run.
        from ._lower_as_pass import apply_lower_as_regions as _apply_lower_as
        _apply_lower_as(exported_program=exported_program, fx_module=_gm)

    # Build annotation IR for any remaining regions.
    _region_table = get_region_table()
    if _region_table:
        gm = exported_program.module()
        build_annotation_ir(exported_program, gm, _region_table)
        _extra_impl = get_impl_registry()
        if _extra_impl:
            _tta = getattr(exported_program, "_tta", {}) or {}
            _tta["impl_registry"] = _extra_impl
            exported_program._tta = _tta

    # Clean up transient TTA node-meta keys.
    for _node in exported_program.graph_module.graph.nodes:
        _node.meta.pop("tta_regions", None)
        _node.meta.pop("tta_seq", None)

    clear_lower_as_regions()
    reset_region_table()


# Register TTA hooks into torch_tensorrt at import time so that
# torch_tensorrt.compile(model, inputs) transparently captures TTA annotations.
from torch_tensorrt.dynamo._compiler import (
    register_export_context as _register_export_context,
    register_post_trace_hook as _register_post_trace_hook,
)
_register_export_context(_tta_export_context)
_register_post_trace_hook(_tta_post_trace_hook)
