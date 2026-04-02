"""
TTA Meta Implementations

Provides metadata inference for plugin and builtin implementations.
"""

from typing import Any, Callable, Dict, Optional

import tensorrt as trt

from ._specs import BuiltinSpec, RegistryPluginSpec


def infer_plugin_metadata(
    spec: RegistryPluginSpec, meta_impl: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Infer metadata for a plugin implementation.

    Attempts metadata resolution in the following order:
      1. If ``meta_impl`` is provided, call it with ``spec`` and return the result.
      2. Otherwise, query the TRT plugin registry for the plugin creator.
         Both the V3 QDP path (``registry.get_creator``) and the classic V2 path
         (``registry.get_plugin_creator``) are tried.
      3. If the registry lookup fails or the plugin is not registered, return basic
         identification metadata with ``found_in_registry=False``.

    Args:
        spec: RegistryPluginSpec describing the plugin (name, version, namespace).
        meta_impl: Optional callable with signature ``(spec: RegistryPluginSpec) -> Dict[str, Any]``
            that returns custom metadata for the plugin.

    Returns:
        Dictionary of metadata containing at minimum ``name``, ``version``,
        ``namespace``, and ``found_in_registry``.

    Raises:
        RuntimeError: If ``meta_impl`` is provided but raises an exception.
    """
    # If user provided meta_impl, use it.
    if meta_impl is not None:
        try:
            return meta_impl(spec)
        except Exception as e:
            raise RuntimeError(
                f"TTA Plugin meta_impl failed for plugin '{spec.namespace}::{spec.name}' "
                f"(version='{spec.version}'): {e}"
            ) from e

    # Try to get metadata from the TRT plugin registry.
    try:
        registry = trt.get_plugin_registry()

        # Try V3 QDP path first (plugins registered via @trtp.register).
        get_creator_fn = getattr(registry, "get_creator", None)
        if get_creator_fn is not None:
            creator = get_creator_fn(spec.name, spec.version, spec.namespace)
            if creator is not None:
                return {
                    "name": spec.name,
                    "version": spec.version,
                    "namespace": spec.namespace,
                    "found_in_registry": True,
                    "creator_version": "v3",
                }

        # Fall back to classic V2 path.
        creator_v2 = registry.get_plugin_creator(spec.name, spec.version, spec.namespace)
        if creator_v2 is not None:
            return {
                "name": spec.name,
                "version": spec.version,
                "namespace": spec.namespace,
                "found_in_registry": True,
                "creator_version": "v2",
            }
    except (AttributeError, RuntimeError):
        # Registry unavailable or the plugin lookup raised an internal TRT error.
        pass

    # Fallback: plugin not found in registry.
    return {
        "name": spec.name,
        "version": spec.version,
        "namespace": spec.namespace,
        "found_in_registry": False,
    }


def infer_builtin_metadata(spec: BuiltinSpec) -> Dict[str, Any]:
    """
    Infer metadata for a builtin TRT layer implementation.

    .. note::
        Full shape/dtype inference via a scratch TRT network is not yet implemented.
        A correct implementation would:

        1. Create a scratch ``trt.Builder`` and ``INetworkDefinition``.
        2. Add placeholder ``ITensor`` inputs with dummy concrete shapes and dtypes
           derived from ``spec.kwargs`` (or caller-supplied tensor metadata).
        3. Call ``getattr(network, spec.add_name)(*placeholder_inputs, **layer_kwargs)``
           to obtain the ``ILayer``.
        4. Extract output metadata (shapes, dtypes) from ``layer.get_output(i)``
           for ``i`` in ``range(layer.num_outputs)``.
        5. Destroy the scratch network and builder.

        This requires concrete input tensor descriptors that are not available at
        spec-definition time; callers that need accurate output-shape information
        must either provide a ``meta_impl`` or perform shape inference downstream
        (e.g. inside the TRTInterpreter).

    Args:
        spec: BuiltinSpec describing the TRT layer (``add_name``, ``kwargs``).

    Raises:
        NotImplementedError: Always.  Callers that require this metadata must
            implement shape inference at the call site using concrete input
            tensor descriptors.
    """
    raise NotImplementedError(
        f"infer_builtin_metadata is not implemented for '{spec.add_name}'. "
        "Accurate output-shape/dtype inference for builtin TRT layers requires "
        "concrete input tensor descriptors (shapes and dtypes) that are not "
        "available at spec-definition time. "
        "Perform shape inference inside the TRTInterpreter using the live "
        "INetworkDefinition, or supply a meta_impl callable that returns the "
        "expected output metadata."
    )


def validate_plugin_metadata(spec: RegistryPluginSpec, metadata: Dict[str, Any]) -> bool:
    """
    Validate plugin metadata against a RegistryPluginSpec.

    Checks that the metadata dictionary contains the minimum required fields
    (``name``, ``version``, ``namespace``) needed to identify the plugin.

    Args:
        spec: RegistryPluginSpec the metadata was derived from.
        metadata: Metadata dictionary to validate.

    Returns:
        ``True`` if all required fields are present, ``False`` otherwise.
    """
    required_fields = ["name", "version", "namespace"]
    return all(field in metadata for field in required_fields)


def validate_builtin_metadata(spec: BuiltinSpec, metadata: Dict[str, Any]) -> bool:
    """
    Validate builtin layer metadata against a BuiltinSpec.

    Checks that the metadata dictionary contains the minimum required field
    (``add_name``) needed to identify the TRT network method.

    Args:
        spec: BuiltinSpec the metadata was derived from.
        metadata: Metadata dictionary to validate.

    Returns:
        ``True`` if all required fields are present, ``False`` otherwise.
    """
    required_fields = ["add_name"]
    return all(field in metadata for field in required_fields)
