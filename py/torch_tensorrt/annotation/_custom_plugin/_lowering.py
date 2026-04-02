"""Custom plugin lowering — bridges KernelImplSpec / CustomPluginSpec to TRT.

Role in the compilation pipeline
---------------------------------
This module is the converter entry-point invoked by ``_torchtrt_integration.py``
when the TTA lowering pass encounters a boundary op whose ``AnnotationMetadata``
carries a ``KernelImplSpec`` or a ``CustomPluginSpec``.

It performs two tasks:

1. **Normalisation** (``_spec_to_descriptor``) — coerces the legacy
   ``KernelImplSpec`` dataclass (used by ``@tta.export_as`` before the
   descriptor API was introduced) into the canonical ``CustomPluginSpec``
   that drives QDP registration.

2. **Lowering** (``lower_custom_plugin``) — delegates to
   ``lower_custom_plugin_descriptor`` which calls ``@trtp.register`` /
   ``@trtp.aot_impl`` as needed and inserts the resulting ``IPluginV3`` layer
   into the active ``INetworkDefinition``.

A third helper (``register_custom_plugin_qdp``) is provided for callers that
need to register a descriptor under an *explicit* name/namespace rather than
the auto-computed fingerprint-derived op name.

Dependencies
------------
- ``tensorrt`` must be importable; the import is mandatory because this module
  is only loaded inside the TRT compilation path.
- ``CustomPluginSpec`` / ``lower_custom_plugin_descriptor`` /
  ``register_custom_plugin`` from ``._descriptor``.
- ``KernelImplSpec`` from ``.._specs`` (legacy spec type).
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Mandatory TRT import — this module is only used inside the TRT compile path
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt
except ImportError as e:
    raise ImportError(
        "TensorRT is required for custom plugin lowering. "
        "Install it with: pip install tensorrt  (or use the TensorRT container). "
        "This module should only be imported inside a TRT compilation context."
    ) from e

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from ._descriptor import (
    CustomPluginSpec,
    custom_plugin as _make_descriptor,
    lower_custom_plugin_descriptor,
    register_custom_plugin,
)
from ._qdp_utils import TTAPluginError
from .._specs import KernelImplSpec

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Accepted spec types for the public lowering API.
_SpecOrDescriptor = Union[KernelImplSpec, CustomPluginSpec]

#: Output type of lowering: a single ITensor or an ordered tuple of ITensors.
_LoweringOutput = Union[trt.ITensor, Tuple[trt.ITensor, ...]]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spec_to_descriptor(spec: _SpecOrDescriptor) -> CustomPluginSpec:
    """Normalise *spec* to a ``CustomPluginSpec``.

    ``CustomPluginSpec`` is returned unchanged.  A ``KernelImplSpec``
    (the legacy dataclass populated by ``@tta.export_as`` before the descriptor
    API was introduced) is converted by re-running the ``custom_plugin``
    factory with its ``kernel`` and ``meta_impl`` fields.

    Args:
        spec: Either a ``CustomPluginSpec`` (new API) or a
            ``KernelImplSpec`` (legacy API).

    Returns:
        A ``CustomPluginSpec`` ready for QDP registration and lowering.

    Raises:
        TypeError: If *spec* is neither a ``CustomPluginSpec`` nor a
            ``KernelImplSpec``.  The error message includes the actual type
            so the caller can diagnose mis-wired converters.
        TTAPluginError: Propagated from ``custom_plugin()`` if the kernel spec
            inside a ``KernelImplSpec`` is invalid or unsupported.
    """
    if isinstance(spec, CustomPluginSpec):
        return spec

    if isinstance(spec, KernelImplSpec):
        # KernelImplSpec → CustomPluginSpec: re-run the factory so that
        # the auto-computed op_name and tactic-table derivation are applied
        # consistently regardless of how the spec was originally constructed.
        return _make_descriptor(kernel=spec.kernel, meta_impl=spec.meta_impl)

    raise TypeError(
        f"_spec_to_descriptor: expected KernelImplSpec or CustomPluginSpec, "
        f"got {type(spec).__qualname__!r}.  Ensure the converter is receiving a "
        f"spec object from tta.custom_plugin() or tta.export_as(impl=...)."
    )


# ---------------------------------------------------------------------------
# Public lowering API
# ---------------------------------------------------------------------------


def lower_custom_plugin(
    ctx: Any,
    spec: _SpecOrDescriptor,
    trt_inputs: List[trt.ITensor],
    name: str,
) -> _LoweringOutput:
    """Lower a custom plugin spec/descriptor to a TRT ``IPluginV3`` layer.

    This is the primary entry-point called by the TTA converter
    (``_torchtrt_integration.py``) when it encounters a boundary op that
    carries a custom-plugin spec.  It normalises *spec* if necessary, then
    delegates all QDP registration and layer insertion to
    ``lower_custom_plugin_descriptor``.

    Args:
        ctx: Torch-TRT ``ConversionContext`` (carries the active
            ``INetworkDefinition`` and build-time state).
        spec: A ``KernelImplSpec`` (legacy) or ``CustomPluginSpec``
            (current API) describing the kernel and its meta-shape function.
        trt_inputs: Ordered list of ``trt.ITensor`` objects corresponding to
            the boundary op's inputs.
        name: Human-readable layer name forwarded to TRT for debugging and
            profiling (visible in engine inspector output).

    Returns:
        A single ``trt.ITensor`` for single-output plugins, or a tuple of
        ``trt.ITensor`` objects for multi-output plugins.

    Raises:
        TypeError: If *spec* is an unrecognised type (propagated from
            ``_spec_to_descriptor``).
        TTAPluginError: If QDP registration fails due to an invalid or
            inconsistent descriptor (propagated from
            ``lower_custom_plugin_descriptor``).
        trt.tensorrt.IBuilderError: If TRT rejects the plugin layer (e.g.
            unsupported dtype/format combination).
    """
    descriptor: CustomPluginSpec = _spec_to_descriptor(spec)
    return lower_custom_plugin_descriptor(ctx, descriptor, trt_inputs, name)


def register_custom_plugin_qdp(
    descriptor: CustomPluginSpec,
    plugin_name: str,
    namespace: str = "tta_custom",
    version: str = "1",
    num_inputs: int = 1,
) -> None:
    """Register a ``CustomPluginSpec`` with QDP under a caller-supplied name.

    Unlike ``register_custom_plugin`` (which uses the auto-computed op name
    derived from the kernel fingerprint), this function registers the plugin
    under an *explicit* ``namespace::plugin_name`` pair, making it retrievable
    via the TRT plugin registry::

        trt.get_plugin_registry().get_creator(plugin_name, "1", namespace)

    The descriptor is shallow-cloned with the new op name before registration
    so the original descriptor is not mutated.

    .. note::
        TRT's QDP framework always uses version ``"1"`` internally.  The
        *version* parameter is accepted for API symmetry with the TRT plugin
        registry lookup signature but does **not** affect the registration.

    Args:
        descriptor: ``CustomPluginSpec`` returned by ``tta.custom_plugin()``.
        plugin_name: Plugin name *without* namespace prefix, e.g. ``"my_kernel"``.
            Must be a valid C identifier (no spaces or special characters).
        namespace: QDP namespace under which to register.  Defaults to
            ``"tta_custom"``.  Use a project-specific namespace to avoid
            collisions in shared processes (e.g. pytest-xdist workers).
        version: Plugin version for TRT registry lookup.  Defaults to ``"1"``.
            Informational only — does not change QDP behaviour.
        num_inputs: Number of tensor inputs expected by the plugin.  Must
            match the arity of the ``@trtp.register`` descriptor function
            inside the ``CustomPluginSpec``.  Defaults to ``1``.

    Returns:
        None.  Registration side-effects are process-global and idempotent
        (a second call with the same op name is a no-op).

    Raises:
        TTAPluginError: If QDP registration fails (e.g. the descriptor's
            kernel spec is invalid or the ``@trtp.register`` callback raises).

    Example::

        spec = tta.custom_plugin(tta.triton(my_kernel))
        register_custom_plugin_qdp(spec, "my_kernel", namespace="my_ns")
        creator = trt.get_plugin_registry().get_creator("my_kernel", "1", "my_ns")
    """
    op_name: str = f"{namespace}::{plugin_name}"

    # Shallow-clone the descriptor with the explicit op name so that the
    # caller's original descriptor (with its auto-computed fingerprint name)
    # is preserved unchanged.
    named_descriptor = CustomPluginSpec(
        op_name=op_name,
        specs=descriptor.specs,
        meta_impl=descriptor.meta_impl,
        attrs=descriptor.attrs,
        weights=descriptor.weights,
        self_attrs=descriptor.self_attrs,
    )
    register_custom_plugin(named_descriptor, num_inputs)
