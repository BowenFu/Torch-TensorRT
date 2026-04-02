"""TTA builtin lowering: converts BuiltinSpec leaf ops to TensorRT built-in layer calls."""

import inspect
import logging
from typing import List, Sequence, Union

import tensorrt as trt

from ._binder import bind_spec_to_layer
from ._errors import TTABuiltinError
from ._layer_metadata import set_tta_layer_metadata
from ._plugin_lowering import _extract_layer_outputs
from ._signature import get_signature_model
from ._specs import BuiltinSpec

logger = logging.getLogger(__name__)


def lower_builtin(
    ctx: "ConversionContext",
    spec: BuiltinSpec,
    trt_inputs: List[trt.ITensor],
    name: str,
    require: bool = False,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    """
    Lower a BuiltinSpec to a TensorRT layer.

    Args:
        ctx: ConversionContext with ctx.net (TRT INetworkDefinition)
        spec: BuiltinSpec with add_name and kwargs
        trt_inputs: Already-converted TRT input tensors
        name: Layer name (used as leaf_op in diagnostics)
        require: If True, errors during lowering always propagate as
            TTABuiltinError.  If False (default), the caller may choose to
            swallow non-fatal errors.  For builtins, errors are structural
            (method missing, wrong args) so they always raise regardless of
            this flag — the parameter is accepted for API consistency.

    Returns:
        TRT output tensor(s)

    Raises:
        TTABuiltinError: If add_* method not found or lowering fails
    """
    add_method_name = spec.add_name

    # Load SignatureModel for target add_* method
    try:
        signature = get_signature_model(spec)
    except ValueError as e:
        raise TTABuiltinError(
            f"Failed to load signature model for '{add_method_name}': {e}. "
            f"Ensure '{add_method_name}' is a valid method on trt.INetworkDefinition.",
            stage="lowering",
            leaf_op=name,
            impl_id=add_method_name,
        ) from e

    # Use binder to partition kwargs into ctor args and setattr list
    try:
        ctor_args, setattr_list = bind_spec_to_layer(spec, signature, trt_inputs)
    except ValueError as e:
        raise TTABuiltinError(
            f"Failed to bind spec to layer for '{add_method_name}': {e}",
            stage="lowering",
            leaf_op=name,
            impl_id=add_method_name,
        ) from e

    # Get the add_* method from network
    if not hasattr(ctx.net, add_method_name):
        # Suggest similar method names for better diagnostics
        available = [
            m for m in dir(ctx.net) if m.startswith("add_") and callable(getattr(ctx.net, m, None))
        ]
        raise TTABuiltinError(
            f"TensorRT network has no method '{add_method_name}'. "
            f"Available add_* methods: {sorted(available)}",
            stage="lowering",
            leaf_op=name,
            impl_id=add_method_name,
        )

    add_method = getattr(ctx.net, add_method_name)

    # Validate that the resolved attribute is callable (guards against TRT
    # property names that shadow the expected method name).
    if not callable(add_method):
        raise TTABuiltinError(
            f"'{add_method_name}' on trt.INetworkDefinition is not callable "
            f"(got {type(add_method).__name__}). "
            f"Check that the BuiltinSpec add_name refers to an add_* method, not a property.",
            stage="lowering",
            leaf_op=name,
            impl_id=add_method_name,
        )

    # Validate the method signature against the arguments we are about to pass so
    # that mismatches are caught with a clear message before TRT raises a cryptic
    # C-extension error.
    try:
        trt_sig = inspect.signature(add_method)
        positional_count = sum(
            1
            for p in trt_sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.name != "self"
        )
        if trt_inputs and len(trt_inputs) > positional_count:
            logger.warning(
                "lower_builtin(%s): passing %d positional inputs but '%s' "
                "accepts at most %d positional parameters; "
                "this may raise a TypeError from TRT.",
                name,
                len(trt_inputs),
                add_method_name,
                positional_count,
            )
    except (ValueError, TypeError):
        # inspect.signature may fail on some C-extension methods; skip validation
        pass

    # add_normalization requires scale.ndim == bias.ndim == input.ndim.
    # Automatically insert IShuffleLayers to broadcast 1-D (or lower-rank)
    # scale/bias tensors up to the input rank so the constraint is always met.
    if add_method_name == "add_normalization" and len(trt_inputs) >= 3:
        input_tensor = trt_inputs[0]
        input_ndim = len(input_tensor.shape)
        trt_inputs = list(trt_inputs)
        for slot, role in [(1, "scale"), (2, "bias")]:
            t = trt_inputs[slot]
            if len(t.shape) < input_ndim:
                extra = input_ndim - len(t.shape)
                shuf = ctx.net.add_shuffle(t)
                shuf.reshape_dims = (1,) * extra + tuple(t.shape)
                shuf.name = f"{name}_{role}_expand"
                trt_inputs[slot] = shuf.get_output(0)

    # Call network.add_<target>(*trt_inputs, **ctor_args)
    try:
        if trt_inputs:
            layer = add_method(*trt_inputs, **ctor_args)
        else:
            layer = add_method(**ctor_args)
    except TypeError as e:
        raise TTABuiltinError(
            f"Wrong arguments calling '{add_method_name}' for layer '{name}': {e}. "
            f"Passed {len(trt_inputs)} positional input(s) and keyword args: "
            f"{sorted(ctor_args)}. "
            f"Check that all required parameters are included in the BuiltinSpec kwargs.",
            stage="lowering",
            leaf_op=name,
            impl_id=add_method_name,
        ) from e
    except RuntimeError as e:
        raise TTABuiltinError(
            f"TensorRT error calling '{add_method_name}' for layer '{name}': {e}",
            stage="lowering",
            leaf_op=name,
            impl_id=add_method_name,
        ) from e

    if layer is None:
        raise TTABuiltinError(
            f"'{add_method_name}' returned None for layer '{name}'. "
            f"This usually indicates a TRT network error (e.g. incompatible input shapes "
            f"or invalid parameter values). Check the TRT error log for details.",
            stage="lowering",
            leaf_op=name,
            impl_id=add_method_name,
        )

    # Apply setattr for leftover kwargs
    for attr_name, attr_value in setattr_list:
        try:
            setattr(layer, attr_name, attr_value)
        except AttributeError as e:
            raise TTABuiltinError(
                f"Layer '{name}' (type {type(layer).__name__}, created by '{add_method_name}'): "
                f"no attribute '{attr_name}'. "
                f"Check that '{attr_name}' is a valid settable attribute for this layer type.",
                stage="lowering",
                leaf_op=name,
                impl_id=add_method_name,
            ) from e
        except (TypeError, ValueError) as e:
            raise TTABuiltinError(
                f"Layer '{name}' (type {type(layer).__name__}, created by '{add_method_name}'): "
                f"cannot set attribute '{attr_name}' to {attr_value!r}: {e}",
                stage="lowering",
                leaf_op=name,
                impl_id=add_method_name,
            ) from e

    layer.name = name
    set_tta_layer_metadata(layer, "builtin", spec.add_name, name)

    return _extract_layer_outputs(layer)
