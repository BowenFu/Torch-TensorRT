"""
TTA Plugin Lowering

Lowers TTA plugin leaf ops to TensorRT plugin layers via the TRT plugin registry.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt

from ._errors import TTAPluginError
from ._layer_metadata import set_tta_layer_metadata
from ._specs import RegistryPluginSpec

logger = logging.getLogger(__name__)


def _extract_layer_outputs(layer):
    """Return single output or tuple of outputs from a TRT plugin/builtin layer."""
    if layer.num_outputs == 1:
        return layer.get_output(0)
    return tuple(layer.get_output(i) for i in range(layer.num_outputs))


def encode_attrs_to_plugin_fields(attrs: Dict[str, Any]) -> list:
    """
    Encode attributes dict to TensorRT PluginField list.

    Args:
        attrs: Dictionary of attribute name -> value

    Returns:
        List of trt.PluginField objects
    """
    fields = []

    for name, value in attrs.items():
        if isinstance(value, bool):
            field = trt.PluginField(
                name, np.array([int(value)], dtype=np.int32), trt.PluginFieldType.INT32
            )
        elif isinstance(value, int):
            field = trt.PluginField(
                name, np.array([value], dtype=np.int32), trt.PluginFieldType.INT32
            )
        elif isinstance(value, float):
            field = trt.PluginField(
                name, np.array([value], dtype=np.float32), trt.PluginFieldType.FLOAT32
            )
        elif isinstance(value, str):
            field = trt.PluginField(
                name, np.frombuffer(value.encode(), dtype=np.uint8), trt.PluginFieldType.CHAR
            )
        elif isinstance(value, (list, tuple)):
            field = trt.PluginField(
                name, np.array(value, dtype=np.int32), trt.PluginFieldType.INT32
            )
        elif isinstance(value, np.ndarray):
            _dtype_map = {
                np.float32: trt.PluginFieldType.FLOAT32,
                np.float64: trt.PluginFieldType.FLOAT64,
                np.int32: trt.PluginFieldType.INT32,
                np.int64: trt.PluginFieldType.INT64,
            }
            pf_type = _dtype_map.get(value.dtype.type)
            if pf_type is None:
                logger.debug(
                    "encode_attrs_to_plugin_fields: unrecognized numpy dtype %s for field %r, "
                    "defaulting to FLOAT32",
                    value.dtype,
                    name,
                )
                pf_type = trt.PluginFieldType.FLOAT32
            field = trt.PluginField(name, value, pf_type)
        elif hasattr(value, "detach") and hasattr(value, "cpu"):
            arr = value.detach().cpu().numpy().astype(np.float32)
            field = trt.PluginField(
                name, arr, trt.PluginFieldType.FLOAT32
            )
        else:
            logger.debug(
                "encode_attrs_to_plugin_fields: skipping field %r with unsupported type %s",
                name,
                type(value).__name__,
            )
            continue

        fields.append(field)

    return fields


def _find_creator_by_name_version(
    name: str, version: str, namespace: str
) -> Any:
    """Scan the plugin registry list for a creator matching (name, version, namespace).

    Fallback for registries that do not expose ``get_creator`` (V3/QDP API).
    Returns the first matching creator object or ``None``.
    """
    registry = trt.get_plugin_registry()
    creator_list = getattr(registry, "plugin_creator_list", None)
    if creator_list is None:
        return None
    for creator in creator_list:
        c_name = getattr(creator, "name", None)
        c_ver = getattr(creator, "plugin_version", None) or getattr(
            creator, "version", None
        )
        c_ns = getattr(creator, "plugin_namespace", None) or ""
        if (
            c_name == name
            and str(c_ver) == str(version)
            and (c_ns or "") == (namespace or "")
        ):
            return creator
    return None


def _try_v3_qdp(
    registry: Any,
    network: Any,
    spec: RegistryPluginSpec,
    field_collection: Any,
    trt_inputs: List[trt.ITensor],
    name: str,
    impl_id: str,
) -> Optional[Any]:
    """Try the V3/QDP code path.  Returns a plugin layer or None.

    Plugins registered via @trtp.register are visible through get_creator (not
    get_plugin_creator).  Raises TTAPluginError on internal failures.
    """
    get_creator_fn = getattr(registry, "get_creator", None)
    creator_v3 = None
    if get_creator_fn is not None:
        creator_v3 = get_creator_fn(spec.name, spec.version, spec.namespace)
    if creator_v3 is None:
        creator_v3 = _find_creator_by_name_version(
            spec.name, spec.version, spec.namespace
        )

    if creator_v3 is None:
        return None

    try:
        phase = getattr(trt.TensorRTPhase, "BUILD", None)
        if phase is None:
            raise TTAPluginError(
                "TensorRTPhase.BUILD not available",
                stage="lowering",
                leaf_op=name,
                impl_id=impl_id,
            )
        plugin = creator_v3.create_plugin(name, field_collection, phase)
        if plugin is None:
            raise TTAPluginError(
                "V3 plugin creator returned None",
                stage="lowering",
                leaf_op=name,
                impl_id=impl_id,
            )
        shape_inputs: List[trt.ITensor] = []
        plugin_layer = network.add_plugin_v3(trt_inputs, shape_inputs, plugin)
        plugin_layer.name = name
        set_tta_layer_metadata(plugin_layer, "plugin", spec.name, name, attrs=spec.attrs or None)
        return plugin_layer
    except TTAPluginError:
        raise
    except Exception as e:
        # Broad catch is necessary: TRT plugin API methods (create_plugin,
        # add_plugin_v3) raise a mix of RuntimeError, AttributeError, and
        # TRT-internal C++ exceptions depending on the failure mode and TRT version.
        raise TTAPluginError(
            f"V3 plugin '{impl_id}' failed during network lowering (op '{name}'): {e}",
            stage="lowering",
            leaf_op=name,
            impl_id=impl_id,
        ) from e


def _try_v2(
    registry: Any,
    network: Any,
    spec: RegistryPluginSpec,
    field_collection: Any,
    trt_inputs: List[trt.ITensor],
    name: str,
    impl_id: str,
) -> Any:
    """Try the V2 code paths (two-arg, then three-arg hybrid).  Returns a plugin layer.

    First attempts the classic two-argument create_plugin / add_plugin_v2 call.
    If that fails, retries with a three-argument create_plugin / add_plugin_v3
    hybrid call (available in newer TRT versions).  Raises TTAPluginError on
    failure or if no V2 creator is found.
    """
    plugin_creator_v2 = registry.get_plugin_creator(
        spec.name, spec.version, spec.namespace
    )

    if plugin_creator_v2 is None:
        raise TTAPluginError(
            f"Plugin not found in TensorRT registry (no V2 or V3 creator): {impl_id}",
            stage="lowering",
            leaf_op=name,
            impl_id=impl_id,
        )

    # First attempt: two-argument create_plugin (classic V2 API).
    v2_exc: Optional[Exception] = None
    try:
        plugin = plugin_creator_v2.create_plugin(name, field_collection)
        if plugin is None:
            raise TTAPluginError(
                "Plugin creator returned None",
                stage="lowering",
                leaf_op=name,
                impl_id=impl_id,
            )
        plugin_layer = network.add_plugin_v2(trt_inputs, plugin)
        plugin_layer.name = name
        set_tta_layer_metadata(plugin_layer, "plugin", spec.name, name, attrs=spec.attrs or None)
        return plugin_layer
    except TTAPluginError:
        raise
    except Exception as e:
        # Broad catch is necessary: TRT V2 create_plugin raises different exception
        # types across versions (RuntimeError, AttributeError, ValueError) and we
        # want to retry the three-arg form before surfacing the error.
        logger.debug(
            "[TTA Plugin lowering] %s: V2 two-arg create_plugin failed (%s); "
            "retrying with phase arg",
            name,
            e,
        )
        v2_exc = e

    # Second attempt: three-argument create_plugin + add_plugin_v3 (hybrid).
    phase = getattr(trt.TensorRTPhase, "BUILD", None)
    if phase is not None:
        try:
            plugin = plugin_creator_v2.create_plugin(name, field_collection, phase)
            if plugin is not None:
                shape_inputs: List[trt.ITensor] = []
                plugin_layer = network.add_plugin_v3(trt_inputs, shape_inputs, plugin)
                plugin_layer.name = name
                set_tta_layer_metadata(plugin_layer, "plugin", spec.name, name, attrs=spec.attrs or None)
                return plugin_layer
        except TTAPluginError:
            raise
        except Exception as e:
            # Broad catch is necessary: same rationale as the two-arg attempt above —
            # TRT API raises heterogeneous exception types depending on failure mode.
            logger.debug(
                "[TTA Plugin lowering] %s: V2 three-arg create_plugin+add_plugin_v3 "
                "also failed (%s)",
                name,
                e,
            )
            raise TTAPluginError(
                f"V2 plugin '{impl_id}' failed during network lowering (op '{name}'): two-arg error: {v2_exc}; three-arg+v3 error: {e}",
                stage="lowering",
                leaf_op=name,
                impl_id=impl_id,
            ) from e

    # phase unavailable and two-arg already failed
    raise TTAPluginError(
        f"V2 plugin failed: {v2_exc}",
        stage="lowering",
        leaf_op=name,
        impl_id=impl_id,
    ) from v2_exc


def lower_plugin(
    ctx: "ConversionContext",
    spec: RegistryPluginSpec,
    trt_inputs: List[trt.ITensor],
    name: str,
    require: bool = True,
) -> Union[trt.ITensor, Sequence[trt.ITensor], None]:
    """
    Lower a RegistryPluginSpec to a TensorRT plugin layer.
    Supports both V2 (IPluginCreator / add_plugin_v2) and V3 (IPluginCreatorV3One / add_plugin_v3) plugins.

    Args:
        ctx: ConversionContext with ctx.net (TRT INetworkDefinition)
        spec: RegistryPluginSpec with name, version, namespace, attrs
        trt_inputs: Already-converted TRT input tensors
        name: Layer name (used as leaf_op in diagnostics)
        require: If True (default), raise TTAPluginError when plugin is not
            found in the registry.  If False, log a warning and return None
            instead of raising, allowing the caller to fall back gracefully.

    Returns:
        TRT output tensor(s), or None if plugin not found and require=False.

    Raises:
        TTAPluginError: If plugin not found or lowering fails (always raised
            when require=True; only for non-"not found" errors when require=False).
    """
    # Compose a human-readable impl_id for diagnostics
    impl_id = f"{spec.name} v{spec.version} ({spec.namespace})"

    plugin_fields = encode_attrs_to_plugin_fields(spec.attrs)
    field_collection = trt.PluginFieldCollection(plugin_fields)
    registry = trt.get_plugin_registry()

    # Try V3/QDP path first, then fall back to V2.
    plugin_layer = _try_v3_qdp(
        registry, ctx.net, spec, field_collection, trt_inputs, name, impl_id
    )
    if plugin_layer is not None:
        return _extract_layer_outputs(plugin_layer)

    # V3 not found — try V2.
    try:
        plugin_layer = _try_v2(
            registry, ctx.net, spec, field_collection, trt_inputs, name, impl_id
        )
        return _extract_layer_outputs(plugin_layer)
    except TTAPluginError as e:
        # Re-raise all non-"not found" errors unconditionally.  For "not found"
        # errors honour the require flag.
        not_found_marker = "no V2 or V3 creator"
        if not_found_marker not in str(e):
            raise
        if require:
            raise
        logger.warning(
            "[TTA Plugin lowering] %s: %s — skipping (require=False)", name, e
        )
        return None
