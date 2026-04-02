"""
TTA Binder

Maps BuiltinSpec kwargs to TensorRT add_* constructor arguments and layer attributes.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt

from ._signature import SignatureModel
from ._specs import BuiltinSpec

logger = logging.getLogger(__name__)


def bind_spec_to_layer(
    spec: BuiltinSpec, signature: SignatureModel, input_tensors: List[trt.ITensor]
) -> Tuple[Dict[str, Any], List[Tuple[str, Any]]]:
    """
    Bind BuiltinSpec kwargs to constructor args and setattr list.

    Args:
        spec: BuiltinSpec with add_name and kwargs
        signature: SignatureModel for the target add_* method
        input_tensors: List of input tensors (already converted to TRT)

    Returns:
        Tuple of (ctor_args dict, setattr_list)
            ctor_args: Dict of parameter names to values for add_* constructor
            setattr_list: List of (attr_name, value) tuples for post-creation setattr
    """
    ctor_args: Dict[str, Any] = {}
    setattr_list: List[Tuple[str, Any]] = []

    # Get param type info from signature (may be None for older SignatureModels)
    param_types = signature.param_types or {}

    # Partition spec.kwargs based on signature
    for key, value in spec.kwargs.items():
        expected_type = param_types.get(key, "")
        if key in signature.ctor_param_names:
            # This kwarg matches a constructor parameter
            converted_value = _convert_value_for_trt(key, value, expected_type, spec.add_name)
            ctor_args[key] = converted_value
        else:
            # This kwarg should be applied via setattr after layer creation
            converted_value = _convert_value_for_trt(key, value, expected_type, spec.add_name)
            setattr_list.append((key, converted_value))

    return ctor_args, setattr_list



# Python built-in type names that are never TRT enum classes.  Skipping them
# prevents spurious stdout prints: TRT's C extension __getattr__ prints
# "AttributeError: <name>" before raising AttributeError when an unknown
# attribute is accessed, and Python's three-arg getattr() silences the
# exception but cannot suppress the side-effect print.
_PYTHON_BUILTIN_NAMES = frozenset(
    {"int", "float", "bool", "str", "bytes", "list", "dict", "tuple", "set",
     "complex", "bytearray", "memoryview", "type", "object"}
)


def _resolve_trt_enum(type_str: str) -> Optional[Any]:
    """
    Resolve a TRT type annotation string to the actual TRT class.

    Args:
        type_str: Type annotation from docstring (e.g., "tensorrt.tensorrt.ActivationType")

    Returns:
        The TRT class/enum, or None if not found
    """
    if not type_str:
        return None

    # Extract the class name from the type string
    # Handles: "tensorrt.tensorrt.ActivationType", "tensorrt.ActivationType", "ActivationType"
    parts = type_str.strip().split(".")
    class_name = parts[-1]

    # Skip Python built-in type names: TRT's C extension prints
    # "AttributeError: <name>" to stdout when getattr misses, and we cannot
    # suppress that side-effect via the default-arg form of getattr.
    if class_name in _PYTHON_BUILTIN_NAMES:
        return None

    # Try to get from trt module
    return getattr(trt, class_name, None)


_DIMS_PARAM_KEYWORDS = frozenset(
    {"shape", "stride", "start", "kernel", "padding", "dilation", "dims"}
)


def _should_convert_to_dims(param_name: str) -> bool:
    """Return True if *param_name* names a parameter that expects a ``trt.Dims`` value.

    Checks whether any of the canonical dimension-related keywords
    (``shape``, ``stride``, ``start``, ``kernel``, ``padding``, ``dilation``,
    ``dims``) appears as a case-insensitive substring of *param_name*.  For
    all other int-list parameters the value is left as a plain Python
    list/tuple so that TRT APIs that expect a sequence rather than ``Dims``
    are not broken.
    """
    lower = param_name.lower()
    return any(kw in lower for kw in _DIMS_PARAM_KEYWORDS)


def _convert_value_for_trt(
    key: str, value: Any, expected_type: str = "", add_name: str = ""
) -> Any:
    """
    Convert Python value to TensorRT-compatible type.

    Args:
        key: Parameter/attribute name (for context in error messages)
        value: Python value to convert
        expected_type: Expected type annotation string from signature
        add_name: Name of the add_* method being lowered (for error messages)

    Returns:
        TRT-compatible value

    Conversions:
        - int + TRT enum type -> trt.EnumType(int)
        - tuple/list of ints -> trt.Dims  (only for dimension-related params;
          see _should_convert_to_dims)
        - numpy array -> trt.Weights
        - tensor -> trt.Weights
        - scalar -> as-is

    Raises:
        ValueError: If the value cannot be converted to the expected TRT enum type
    """
    # Convert int to TRT enum if expected type is a TRT enum class.
    # Guard with isinstance(trt_class, type) to skip enum *instances* that TRT
    # exposes as module-level aliases (e.g. trt.bool = DataType.BOOL).
    if isinstance(value, int) and expected_type:
        trt_class = _resolve_trt_enum(expected_type)
        if trt_class is not None and trt_class is not int and isinstance(trt_class, type):
            try:
                return trt_class(value)
            except (ValueError, TypeError) as exc:
                layer_ctx = f" for layer '{add_name}'" if add_name else ""
                raise ValueError(
                    f"Parameter '{key}'{layer_ctx}: cannot convert value {value!r} to "
                    f"TRT enum type '{expected_type}'. "
                    f"Valid values are: {list(trt_class.__members__) if hasattr(trt_class, '__members__') else '<unknown>'}"
                ) from exc

    # Convert tuple/list to Dims only for dimension-related parameters.
    if isinstance(value, (tuple, list)):
        # Check if all elements are integers
        if all(isinstance(v, int) for v in value):
            if _should_convert_to_dims(key):
                return trt.Dims(value)
            # For other int-list params leave as a plain Python list.
            return list(value)
        else:
            # Mixed types or non-int - return as-is
            return value

    # Convert numpy array to Weights
    if isinstance(value, np.ndarray):
        return trt.Weights(value)

    # Convert torch tensor to Weights
    try:
        import torch

        if isinstance(value, torch.Tensor):
            # Convert to numpy first
            np_array = value.detach().cpu().numpy()
            return trt.Weights(np_array)
    except ImportError:
        pass

    # Convert scalar to Weights when the expected parameter type is trt.Weights.
    # TRT methods like add_scale() require Weights for shift/scale/power even
    # when the user passes a plain Python float or int.
    if isinstance(value, (int, float)) and "Weights" in expected_type:
        return trt.Weights(np.array([value], dtype=np.float32))

    # Scalar or other type - return as-is
    return value


def extract_builtin_spec_from_node(target: Any, kwargs: Dict[str, Any]) -> BuiltinSpec:
    """
    Reconstruct BuiltinSpec from FX node information.

    Args:
        target: FX node target (qualified op name string or callable)
        kwargs: FX node kwargs dict, expected to contain a ``spec_kwargs`` entry

    Returns:
        Reconstructed BuiltinSpec

    Note: This function is used during lowering to reconstruct the spec
    from the leaf op node metadata.
    """
    # Extract target name from qualified op name
    # e.g., torch.ops.torch_tensorrt_anno_builtin.builtin_convolution_nd_abc123
    target_str = str(target)

    # Parse to get builtin add_name
    # Format: "builtin_{add_name}_{cache_key}"
    # e.g., "builtin_add_convolution_nd_abc123"
    if "builtin_" in target_str:
        # Extract add_name after "builtin_" prefix
        parts = target_str.split("builtin_")
        if len(parts) > 1:
            # Get everything between "builtin_" and the last underscore (cache key)
            remainder = parts[1]
            # The cache key is the last 16 hex chars, so split from the right
            # Find the last underscore that separates add_name from cache_key
            underscore_idx = remainder.rfind("_")
            if underscore_idx > 0:
                add_name = remainder[:underscore_idx]
            else:
                add_name = remainder
        else:
            add_name = "unknown"
    else:
        add_name = "unknown"

    # Reconstruct spec from kwargs (should contain spec info)
    spec_kwargs = kwargs.get("spec_kwargs", {})

    return BuiltinSpec(add_name=add_name, kwargs=spec_kwargs)
