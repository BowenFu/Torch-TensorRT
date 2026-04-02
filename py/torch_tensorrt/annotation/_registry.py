"""
TTA Op Registry
===============
Handles dynamic registration of TensorRT annotation leaf ops using torch.library.
Each op gets a spec stored in a side table and a per-op Dynamo converter registered
so that torch_tensorrt.dynamo.compile() can find and lower them.

Threading model and lock contract
----------------------------------
All process-global caches in this module are protected by a single reentrant-safe
``_registry_lock`` (a plain ``threading.Lock``).  The locking discipline is:

1. **Op caches** (``_plugin_op_cache``, ``_builtin_op_cache``,
   ``_custom_plugin_op_cache``): written only inside ``_get_or_create_op``.
   The function uses a double-checked locking pattern:

   * Fast path – acquire the lock, check the cache key, and return immediately
     if the op is already registered.
   * Slow path – release the lock, call ``register_tta_leaf_op`` and
     ``_register_converter_for_op`` *without* the lock (both are idempotent
     and the latter performs lazy imports that may re-enter Python's import
     machinery), then re-acquire the lock to write back.

   Invariants:
   - A cache key maps to exactly one ``OpOverload`` object for the lifetime of
     the process; once written it is never mutated or deleted.
   - If two threads race on the same key, both will register (idempotent) and
     the second write-back is silently skipped; callers receive whichever
     ``OpOverload`` was stored first — both are functionally equivalent.

2. **Side tables** (``_spec_registry``, ``_require_registry``,
   ``_name_registry``): written under the lock; read without the lock.
   Reads are safe under CPython's GIL (dict.get is atomic), and these tables
   are only read *after* the corresponding op has been registered (so the key
   always exists when read).  If future ports to a free-threaded Python
   interpreter are needed, reads should also acquire the lock.

3. The lock is **not** held across ``torch.library`` calls or imports to
   prevent potential deadlocks with Python's import lock.
"""

import hashlib
import logging
import threading
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch._ops import OpOverload

from ._custom_plugin._descriptor import CustomPluginSpec
from ._specs import BuiltinSpec, KernelImplSpec, RegistryPluginSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------
_NS_PLUGIN = "torch_tensorrt_anno_plugin"
_NS_BUILTIN = "torch_tensorrt_anno_builtin"
_NS_CUSTOM = "torch_tensorrt_anno_custom_plugin"

# ---------------------------------------------------------------------------
# Process-global caches  (all guarded by _registry_lock — see module docstring)
# ---------------------------------------------------------------------------

# Op caches: cache_key -> registered OpOverload
_plugin_op_cache: Dict[str, OpOverload] = {}
_builtin_op_cache: Dict[str, OpOverload] = {}
_custom_plugin_op_cache: Dict[str, OpOverload] = {}

# Spec side table: OpOverload -> spec that created it
_spec_registry: Dict[
    OpOverload, Union[RegistryPluginSpec, BuiltinSpec, KernelImplSpec, CustomPluginSpec]
] = {}

# Require side table: OpOverload -> require flag
_require_registry: Dict[OpOverload, bool] = {}

# Name side table: OpOverload -> user-supplied annotation name (None = unnamed)
_name_registry: Dict[OpOverload, Optional[str]] = {}

# Single module-level lock protecting all caches above.  See module docstring
# for the full locking discipline.
_registry_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _register_converter_for_op(
    op_overload: OpOverload,
    spec: Union[RegistryPluginSpec, BuiltinSpec, KernelImplSpec, CustomPluginSpec],
) -> None:
    """Register a Dynamo converter for *op_overload* via a lazy import.

    The import is deferred so that importing this module does not hard-require
    Dynamo or TensorRT.  Called outside ``_registry_lock`` to avoid deadlock
    with Python's import machinery.

    Args:
        op_overload: The ``OpOverload`` to register a converter for.
        spec:        The implementation spec associated with the op.
    """
    try:
        from ._torchtrt_integration import register_tta_converter

        register_tta_converter(op_overload, spec)
    except ImportError:
        logger.debug(
            "Dynamo/TRT not available — skipping converter registration for %s",
            op_overload,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_cache_key(
    impl_spec: Union[RegistryPluginSpec, BuiltinSpec, KernelImplSpec],
    runtime_input_arity: int,
) -> str:
    """Compute a deterministic, content-addressed cache key for op registration.

    The key encodes both the implementation spec (via its own
    ``to_cache_key()`` method) and the number of runtime tensor inputs, so two
    specs that differ only in arity get distinct keys.

    Args:
        impl_spec:            Implementation spec (``RegistryPluginSpec``,
                              ``BuiltinSpec``, or ``KernelImplSpec``).
        runtime_input_arity:  Number of runtime tensor inputs.

    Returns:
        A 16-character hexadecimal string derived from a SHA-256 digest.
    """
    spec_key = impl_spec.to_cache_key()
    full_key = (spec_key, runtime_input_arity)
    key_str = str(full_key)
    hash_obj = hashlib.sha256(key_str.encode())
    return hash_obj.hexdigest()[:16]


def generate_schema(input_arity: int, num_outputs: int = 1) -> str:
    """Generate an op schema string for the given input arity and output count.

    Examples::

        generate_schema(2)      # "(Tensor x0, Tensor x1) -> Tensor"
        generate_schema(1, 2)   # "(Tensor x0) -> (Tensor, Tensor)"
        generate_schema(0)      # "() -> Tensor"

    Args:
        input_arity:  Number of tensor inputs.
        num_outputs:  Number of tensor outputs (default ``1``).

    Returns:
        A schema string compatible with ``torch.library.define``.
    """
    tensor_args = ", ".join([f"Tensor x{i}" for i in range(input_arity)])
    inputs = f"({tensor_args})" if input_arity > 0 else "()"
    if num_outputs == 1:
        return f"{inputs} -> Tensor"
    out_types = ", ".join(["Tensor"] * num_outputs)
    return f"{inputs} -> ({out_types})"


def register_tta_leaf_op(
    namespace: str,
    op_name: str,
    schema: str,
    num_outputs: int = 1,
    meta_impl: Optional[Callable[..., Any]] = None,
) -> OpOverload:
    """Register a TTA boundary op via ``torch.library``, or return the existing one.

    This function is **idempotent**: if ``{namespace}::{op_name}`` is already
    registered in ``torch.ops`` it is returned immediately without re-defining
    or re-registering anything.

    The abstract (fake/meta) implementation registered here is used by
    ``torch.export`` and ``torch.compile`` during tracing.  For single-output
    ops it returns a tensor with the same shape/dtype as the first input.  For
    multi-output ops it delegates to *meta_impl* when provided; if *meta_impl*
    raises ``TypeError``, ``RuntimeError``, or ``AttributeError``, or is not
    given, the fallback returns a tuple of clones of the first input.

    Args:
        namespace:   Op namespace (e.g., ``"torch_tensorrt_anno_plugin"``).
        op_name:     Unique op identifier within the namespace.
        schema:      Op schema string (see ``generate_schema``).
        num_outputs: Number of output tensors (default ``1``).
        meta_impl:   Optional callable used by the abstract impl for accurate
                     output shapes/dtypes when *num_outputs* > 1.  Called with
                     the same positional args as the op.

    Returns:
        The registered ``OpOverload`` (the ``default`` overload).
    """
    qualified_name = f"{namespace}::{op_name}"

    # Return the existing op if already registered.
    try:
        return getattr(getattr(torch.ops, namespace), op_name)
    except AttributeError:
        # Op does not exist yet — fall through to register it.
        pass

    torch.library.define(qualified_name, schema, tags=torch.Tag.pt2_compliant_tag)

    _num_outputs = num_outputs
    _meta_impl = meta_impl

    @torch.library.register_fake(qualified_name)
    def abstract_impl(*args: Any, **kwargs: Any) -> Any:
        if _num_outputs == 1:
            return torch.empty(1) if len(args) == 0 else torch.empty_like(args[0])
        # Multi-output: delegate to meta_impl for accurate shapes/dtypes.
        if _meta_impl is not None:
            try:
                outs = _meta_impl(*args)
                if not isinstance(outs, (tuple, list)):
                    outs = (outs,)
                return tuple(outs)
            except (TypeError, RuntimeError, AttributeError) as exc:
                logger.debug(
                    "meta_impl for %s raised %s — falling back to clone-based shapes",
                    qualified_name,
                    exc,
                )
        # Fallback: return a tuple of empty tensors shaped like the first input.
        base = torch.empty(1) if len(args) == 0 else torch.empty_like(args[0])
        return tuple(base.clone() for _ in range(_num_outputs))

    return getattr(getattr(torch.ops, namespace), op_name)


def _get_or_create_op(
    cache: Dict[str, OpOverload],
    namespace: str,
    op_name: str,
    cache_key: str,
    schema: str,
    num_outputs: int,
    meta_impl: Optional[Callable[..., Any]],
    impl_spec: Union[RegistryPluginSpec, BuiltinSpec, KernelImplSpec, CustomPluginSpec],
) -> OpOverload:
    """Shared double-checked-lock pattern for all ``get_or_create_*`` functions.

    Locking discipline (see module docstring for full details):

    1. Acquire the lock and return immediately if *cache_key* is already
       populated (fast path).
    2. Release the lock, register the op and its converter outside the lock
       (slow path — both operations are idempotent and may trigger imports).
    3. Re-acquire the lock to write the result back.  If another thread
       populated the same key concurrently, that entry is kept and the
       current result is discarded (both are equivalent ``OpOverload``
       objects for the same underlying op).

    The returned value is always an ``OpOverload`` (the ``.default`` overload),
    never an ``OpOverloadPacket``.  Dynamo's converter table is keyed on
    ``OpOverload``; storing a packet would cause the partitioner to fail to
    match the op (manifesting as "0 supported operations").

    Args:
        cache:       Per-type op cache to check and write into.
        namespace:   Op namespace string (one of the ``_NS_*`` constants).
        op_name:     Unique op identifier, including any prefix/hash suffix.
        cache_key:   Key used to look up and store the op in *cache*.
        schema:      Op schema string.
        num_outputs: Number of output tensors.
        meta_impl:   Optional abstract-impl callable for multi-output ops.
        impl_spec:   Implementation spec stored in ``_spec_registry``.

    Returns:
        The registered ``OpOverload`` for ``{namespace}::{op_name}``.
    """
    # Fast path — check cache under lock.
    with _registry_lock:
        if cache_key in cache:
            return cache[cache_key]

    # Slow path — register outside the lock (idempotent, may trigger imports).
    op = register_tta_leaf_op(
        namespace, op_name, schema, num_outputs=num_outputs, meta_impl=meta_impl
    )
    op_overload: OpOverload = getattr(op, "default", op)
    _register_converter_for_op(op_overload, impl_spec)

    # Write back under lock; keep whichever entry won the race.
    with _registry_lock:
        if cache_key not in cache:
            cache[cache_key] = op_overload
            _spec_registry[op_overload] = impl_spec
        return cache[cache_key]


def get_or_create_plugin_op(
    impl_spec: RegistryPluginSpec,
    runtime_input_arity: int,
) -> OpOverload:
    """Get or create a TRT-plugin leaf op for the given spec and input arity.

    Args:
        impl_spec:            ``RegistryPluginSpec`` describing the TensorRT plugin.
        runtime_input_arity:  Number of runtime tensor inputs.

    Returns:
        The registered ``OpOverload`` in the ``torch_tensorrt_anno_plugin``
        namespace.
    """
    cache_key = compute_cache_key(impl_spec, runtime_input_arity)
    op_name = f"plugin_{cache_key}"
    schema = generate_schema(runtime_input_arity)
    return _get_or_create_op(
        _plugin_op_cache, _NS_PLUGIN, op_name, cache_key, schema, 1, None, impl_spec
    )


def get_or_create_builtin_op(
    impl_spec: BuiltinSpec,
    runtime_input_arity: int,
) -> OpOverload:
    """Get or create a TRT-builtin leaf op for the given spec and input arity.

    The op name embeds ``impl_spec.add_name`` to make it human-readable in
    FX graph printouts (e.g., ``builtin_add_convolution_nd_<hash>``).

    Args:
        impl_spec:            ``BuiltinSpec`` describing the TRT built-in layer.
        runtime_input_arity:  Number of runtime tensor inputs.

    Returns:
        The registered ``OpOverload`` in the ``torch_tensorrt_anno_builtin``
        namespace.
    """
    cache_key = compute_cache_key(impl_spec, runtime_input_arity)
    op_name = f"builtin_{impl_spec.add_name}_{cache_key}"
    schema = generate_schema(runtime_input_arity)
    return _get_or_create_op(
        _builtin_op_cache, _NS_BUILTIN, op_name, cache_key, schema, 1, None, impl_spec
    )


def get_or_create_custom_plugin_op(
    impl_spec: KernelImplSpec,
    runtime_input_arity: int,
) -> OpOverload:
    """Get or create a custom-plugin leaf op for the given spec and input arity.

    Unlike ``get_or_create_plugin_impl_descriptor_op``, this path is used for
    ``KernelImplSpec`` instances (inline kernel specs) rather than
    ``CustomPluginSpec`` objects.

    Args:
        impl_spec:            ``KernelImplSpec`` describing the kernel.
        runtime_input_arity:  Number of runtime tensor inputs.

    Returns:
        The registered ``OpOverload`` in the
        ``torch_tensorrt_anno_custom_plugin`` namespace.
    """
    cache_key = compute_cache_key(impl_spec, runtime_input_arity)
    op_name = f"custom_plugin_{cache_key}"
    schema = generate_schema(runtime_input_arity)
    return _get_or_create_op(
        _custom_plugin_op_cache, _NS_CUSTOM, op_name, cache_key, schema, 1, None, impl_spec
    )


def get_or_create_plugin_impl_descriptor_op(
    impl_spec: CustomPluginSpec,
    runtime_input_arity: int,
) -> OpOverload:
    """Get or create a custom-plugin leaf op for a ``CustomPluginSpec``.

    ``CustomPluginSpec`` carries a pre-computed, globally unique
    ``op_name`` (set at ``tta.custom_plugin()`` call time, before any FX
    tracing), so the cache key is derived from that name rather than a hash
    of the full spec.  The number of outputs is also pre-computed in the
    descriptor to avoid recording dummy tensor operations into the FX graph
    during tracing.

    Args:
        impl_spec:            ``CustomPluginSpec`` from ``tta.custom_plugin()``.
        runtime_input_arity:  Number of runtime tensor inputs.

    Returns:
        The registered ``OpOverload`` in the
        ``torch_tensorrt_anno_custom_plugin`` namespace.
    """
    num_outputs = impl_spec.num_outputs

    # Use the descriptor's own op_name (already unique) as the basis for the
    # cache key rather than re-hashing the full spec.
    cache_key = f"{impl_spec.op_name}:{runtime_input_arity}:{num_outputs}"
    short_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    op_name = f"custom_plugin_{short_hash}"
    schema = generate_schema(runtime_input_arity, num_outputs)
    return _get_or_create_op(
        _custom_plugin_op_cache,
        _NS_CUSTOM,
        op_name,
        cache_key,
        schema,
        num_outputs,
        impl_spec.meta_impl,
        impl_spec,
    )


# ---------------------------------------------------------------------------
# Side-table accessors
# ---------------------------------------------------------------------------


def get_require_for_op(op_overload: OpOverload) -> bool:
    """Return the ``require`` flag for a registered TTA leaf op.

    Reads are performed without acquiring ``_registry_lock``.  This is safe
    under CPython's GIL because ``dict.get`` is atomic, and this function is
    only called after the op has been fully registered (so the key is
    guaranteed to exist if it was ever set).

    Args:
        op_overload: The ``OpOverload`` to query.

    Returns:
        The ``require`` flag, or ``False`` if the op is not in the registry.
    """
    return _require_registry.get(op_overload, False)


def set_require_for_op(op_overload: OpOverload, require: bool) -> None:
    """Store the ``require`` flag for a TTA leaf op overload.

    Args:
        op_overload: The ``OpOverload`` to update.
        require:     ``True`` if the annotation is required to be lowered.
    """
    with _registry_lock:
        _require_registry[op_overload] = require


def get_name_for_op(op_overload: OpOverload) -> Optional[str]:
    """Return the user-supplied annotation name for a TTA leaf op.

    Reads are performed without acquiring ``_registry_lock`` for the same
    reason as ``get_require_for_op`` — the read is atomic under the GIL and
    only occurs after registration.

    Args:
        op_overload: The ``OpOverload`` to query.

    Returns:
        The annotation name string, or ``None`` if the op has no name.
    """
    return _name_registry.get(op_overload)


def set_name_for_op(op_overload: OpOverload, name: Optional[str]) -> None:
    """Store the user-supplied annotation name for a TTA leaf op overload.

    Passing ``None`` explicitly clears any previously stored name (the entry
    is set to ``None`` in the registry rather than being removed, so that
    ``get_name_for_op`` can distinguish "never named" from "named then
    cleared" — both return ``None`` but subsequent ``set_name_for_op`` calls
    on an already-registered op will overwrite the entry).

    Args:
        op_overload: The ``OpOverload`` to update.
        name:        Annotation name string, or ``None`` to clear.
    """
    with _registry_lock:
        _name_registry[op_overload] = name


# ---------------------------------------------------------------------------
# Primary dispatch entry point
# ---------------------------------------------------------------------------


def get_or_create_op_for_boundary(
    impl_spec: Union[RegistryPluginSpec, BuiltinSpec, KernelImplSpec, CustomPluginSpec],
    runtime_input_arity: int,
) -> OpOverload:
    """Dispatch to the appropriate ``get_or_create_*`` function for *impl_spec*.

    This is the single entry point used by the capture/boundary-insertion
    machinery.  It selects the correct op cache and namespace based on the
    runtime type of *impl_spec*.

    Args:
        impl_spec:            Implementation spec.  Must be one of
                              ``CustomPluginSpec``, ``RegistryPluginSpec``,
                              ``BuiltinSpec``, or ``KernelImplSpec``.
        runtime_input_arity:  Number of runtime tensor inputs at the boundary.

    Returns:
        The registered ``OpOverload`` for use as an FX node target.

    Raises:
        TypeError: If *impl_spec* is not a recognised spec type.  The error
                   message includes the fully-qualified type name to aid
                   debugging.
    """
    if isinstance(impl_spec, CustomPluginSpec):
        return get_or_create_plugin_impl_descriptor_op(impl_spec, runtime_input_arity)
    if isinstance(impl_spec, RegistryPluginSpec):
        return get_or_create_plugin_op(impl_spec, runtime_input_arity)
    if isinstance(impl_spec, BuiltinSpec):
        return get_or_create_builtin_op(impl_spec, runtime_input_arity)
    if isinstance(impl_spec, KernelImplSpec):
        return get_or_create_custom_plugin_op(impl_spec, runtime_input_arity)

    raise TypeError(
        f"get_or_create_op_for_boundary: unrecognised impl_spec type "
        f"'{type(impl_spec).__qualname__}' (module: {type(impl_spec).__module__!r}). "
        f"Expected one of: CustomPluginSpec, RegistryPluginSpec, BuiltinSpec."
    )
