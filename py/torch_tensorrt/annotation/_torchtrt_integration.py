"""
TTA Torch-TensorRT Integration
================================

This module is the bridge between the TTA annotation layer and
torch-tensorrt's Dynamo converter registry.  It is the only place in TTA
that writes into ``DYNAMO_ATEN_CONVERTERS``, the global dict that
``TRTInterpreter`` consults when it encounters an FX node during Dynamo
lowering.

How a TTA leaf op flows through the system
-------------------------------------------
1. A user annotates a function with ``@tta.export_as(impl=...)``.
2. At ``torch.export`` / ``torch.compile`` time the function body is
   replaced by a single *leaf op* registered via ``torch.library`` in
   ``_registry.py`` (one of the ``torch_tensorrt_anno_*`` namespaces).
3. ``_registry._register_converter_for_op`` calls
   ``register_tta_converter`` (this module) immediately after the op is
   registered, so the converter is in place before any compile call.
4. When ``torch_tensorrt.dynamo.compile()`` partitions and lowers the
   exported graph, ``TRTInterpreter`` finds the leaf-op node, looks up
   the converter in ``DYNAMO_ATEN_CONVERTERS``, and calls it.
5. The converter (``_tta_converter`` closure) translates FX args to TRT
   tensors via ``get_trt_tensor``, then dispatches to the appropriate
   lowering helper based on the spec type:

   - ``BuiltinSpec``        → ``_builtin_lowering.lower_builtin``
     (calls ``INetworkDefinition.add_*`` directly)
   - ``RegistryPluginSpec``         → ``_plugin_lowering.lower_plugin``
     (looks up a pre-registered TRT plugin creator)
   - ``CustomPluginSpec`` → ``_custom_plugin._lowering.lower_custom_plugin``
     (instantiates a QDP plugin inline)

6. The ``require`` flag and the user-supplied annotation name are
   re-read from the side tables in ``_registry`` at converter call time
   (not at registration time) so that late calls to ``export_as()`` that
   update those tables are respected by the already-registered converter.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

from torch._ops import OpOverload
from torch.fx.node import Argument, Target

from ._custom_plugin._descriptor import CustomPluginSpec
from ._specs import BuiltinSpec, KernelImplSpec, RegistryPluginSpec

logger = logging.getLogger(__name__)

# Check runtime availability
try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        DYNAMO_ATEN_CONVERTERS,
        ConverterSupport,
    )
    from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False


def register_tta_converter(
    op_overload: OpOverload,
    spec: Union[RegistryPluginSpec, BuiltinSpec, KernelImplSpec, CustomPluginSpec],
) -> None:
    """
    Register a Dynamo converter for a specific TTA leaf op overload.

    This is called from ``_registry._register_converter_for_op`` immediately
    after a new leaf op is created so that the converter is installed before
    any ``torch_tensorrt.dynamo.compile`` call takes place.

    The converter closure:
      1. Re-reads the ``require`` flag and annotation name from the side
         tables in ``_registry`` at call time (not at registration time) so
         that subsequent ``export_as()`` calls that update those tables are
         reflected by the already-installed converter.
      2. Converts every FX positional argument to a TRT tensor via
         ``get_trt_tensor``.
      3. Dispatches to the appropriate lowering helper:
         - ``BuiltinSpec``               → ``lower_builtin``
         - ``RegistryPluginSpec``                → ``lower_plugin``
         - ``CustomPluginSpec``                      → ``lower_custom_plugin``

    If either ``tensorrt`` or ``torch_tensorrt.dynamo`` is unavailable at
    import time, this function is a no-op (the converter is simply not
    registered; the op will not be partitioned into a TRT sub-graph).

    Args:
        op_overload: The ``OpOverload`` to register a converter for
            (e.g. ``torch.ops.torch_tensorrt_anno_plugin.plugin_abc123.default``).
        spec: The implementation spec that describes how to lower this op.
    """
    if not DYNAMO_AVAILABLE or not TRT_AVAILABLE:
        return

    # Capture in a local so the closure does not hold a reference to the
    # mutable outer variable after the function returns.
    _captured_op_overload = op_overload

    def _tta_converter(
        ctx: "ConversionContext",
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union["trt.ITensor", Sequence["trt.ITensor"]]:
        from ._registry import get_require_for_op

        require: bool = get_require_for_op(_captured_op_overload)
        # Read the user-supplied annotation name from node.meta["layer_metadata"].
        # Set at export_as tracing time (export_as wrapper) and at FX rewrite time
        # (lower_as / autotune passes).  Falls back to the auto-generated FX node name.
        _cur = getattr(ctx, "current_node", None)
        _meta_name: Optional[str] = (
            _cur.meta.get("layer_metadata") if _cur is not None and _cur.meta else None
        )
        trt_layer_name: str = _meta_name or name

        # Convert FX positional args to TRT tensors.
        trt_inputs: List["trt.ITensor"] = [
            get_trt_tensor(ctx, arg, f"{trt_layer_name}_input_{i}")
            for i, arg in enumerate(args)
        ]

        if isinstance(spec, BuiltinSpec):
            from ._builtin_lowering import lower_builtin

            return lower_builtin(ctx, spec, trt_inputs, trt_layer_name, require=require)

        if isinstance(spec, RegistryPluginSpec):
            from ._plugin_lowering import lower_plugin

            return lower_plugin(ctx, spec, trt_inputs, trt_layer_name, require=require)

        if isinstance(spec, (KernelImplSpec, CustomPluginSpec)):
            from ._custom_plugin._lowering import lower_custom_plugin

            return lower_custom_plugin(ctx, spec, trt_inputs, trt_layer_name)

        raise TypeError(
            f"[TTA] No TRT lowering path for spec type {type(spec).__name__!r} "
            f"on op '{op_overload}'. Expected one of: BuiltinSpec, RegistryPluginSpec, "
            f"CustomPluginSpec."
        )

    converter_support = ConverterSupport(
        converter_implementation=_tta_converter,
        supports_dynamic_shapes=True,
    )
    DYNAMO_ATEN_CONVERTERS[op_overload] = [converter_support]
    logger.debug(
        "Registered TTA converter for %s (spec=%s)", op_overload, type(spec).__name__
    )
