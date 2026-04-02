"""
Stage 2f: Custom plugin implementation - AOT dispatch and QDP integration.

Handles QDP op registration, tactic management, and AOT dispatch to backend compilers.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ._aot._cutedsl import compile_cutedsl_kernel
from ._aot._cutile import compile_cutile_program
from ._qdp_utils import AOTMetadata, TTAPluginError
from .._specs import CuTeDSLSpec, CuTileSpec, KernelImplSpec, TritonSpec
from ._aot._triton import compile_triton_kernel


class CustomPluginTacticManager:
    """
    Manages tactics (compiled kernels) for custom plugins.

    One tactic per (spec, config) pair. Caches AOT metadata for reuse.
    """

    def __init__(self):
        """Initialize tactic manager with empty cache."""
        self.cache: Dict[Tuple[Any, ...], AOTMetadata] = {}

    def get_or_compile_tactic(
        self,
        spec: Union[TritonSpec, CuTileSpec, CuTeDSLSpec],
        config: Dict[str, Any],
    ) -> AOTMetadata:
        """
        Get or compile a tactic for (spec, config).

        Args:
            spec: Kernel spec (TritonSpec, CuTileSpec, or CuTeDSLSpec)
            config: Configuration dict

        Returns:
            AOT metadata for this tactic

        Raises:
            TTAPluginError: If compilation fails or spec type is unknown
        """
        # Generate cache key
        cache_key = self._make_cache_key(spec, config)

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Dispatch to appropriate AOT compiler
        metadata = self._dispatch_aot_compile(spec, config)

        # Store in cache
        self.cache[cache_key] = metadata

        return metadata

    def register_tactics_for_custom_plugin(
        self,
        custom_spec: KernelImplSpec,
    ) -> List[AOTMetadata]:
        """
        Register all tactics for a KernelImplSpec.

        Compiles all (spec, config) pairs and returns metadata list.

        Args:
            custom_spec: KernelImplSpec (single or list of kernel specs)

        Returns:
            List of AOT metadata (one per tactic)
        """
        tactics = []

        # Get kernel specs (single or list)
        kernel_specs = (
            custom_spec.kernel
            if isinstance(custom_spec.kernel, list)
            else [custom_spec.kernel]
        )

        for kernel_spec in kernel_specs:
            # Get configs for this spec
            configs = kernel_spec.configs if kernel_spec.configs else [{}]

            for config in configs:
                # Compile tactic
                metadata = self.get_or_compile_tactic(kernel_spec, config)
                tactics.append(metadata)

        return tactics

    def _dispatch_aot_compile(
        self,
        spec: Union[TritonSpec, CuTileSpec, CuTeDSLSpec],
        config: Dict[str, Any],
    ) -> AOTMetadata:
        """
        Dispatch to appropriate AOT compiler based on spec type.

        Args:
            spec: Kernel spec
            config: Configuration dict

        Returns:
            AOT metadata

        Raises:
            TTAPluginError: If spec type is unknown
        """
        if isinstance(spec, TritonSpec):
            return compile_triton_kernel(spec, config)
        elif isinstance(spec, CuTileSpec):
            return compile_cutile_program(spec, config)
        elif isinstance(spec, CuTeDSLSpec):
            return compile_cutedsl_kernel(spec, config)
        else:
            raise TTAPluginError(
                f"Unknown kernel spec type '{type(spec).__name__}' — expected TritonSpec, CuTileSpec, or CuTeDSLSpec",
                stage="aot_impl",
                backend="unknown",
            )

    def _make_cache_key(
        self,
        spec: Union[TritonSpec, CuTileSpec, CuTeDSLSpec],
        config: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """
        Generate cache key for (spec, config).

        Args:
            spec: Kernel spec
            config: Configuration dict

        Returns:
            Hashable cache key tuple
        """
        # Use spec's to_cache_key method if available
        if hasattr(spec, "to_cache_key"):
            return spec.to_cache_key()

        # Fallback: use type name, function id, and config
        config_key = tuple(sorted(config.items())) if config else ()
        return (type(spec).__name__.lower(), id(spec.launch_fn), config_key)


# Global tactic manager instance
_tactic_manager = CustomPluginTacticManager()


def get_tactic_manager() -> CustomPluginTacticManager:
    """
    Get global custom plugin tactic manager.

    Returns:
        CustomPluginTacticManager singleton
    """
    return _tactic_manager


def compile_custom_plugin(
    custom_spec: KernelImplSpec,
) -> List[AOTMetadata]:
    """
    Compile all tactics for a custom plugin spec.

    Args:
        custom_spec: KernelImplSpec to compile

    Returns:
        List of AOT metadata (one per tactic)

    Raises:
        TTAPluginError: If compilation fails
    """
    return _tactic_manager.register_tactics_for_custom_plugin(custom_spec)


def get_aot_metadata(
    spec: Union[TritonSpec, CuTileSpec, CuTeDSLSpec],
    config: Dict[str, Any],
) -> Optional[AOTMetadata]:
    """
    Get cached AOT metadata for (spec, config) if available.

    Args:
        spec: Kernel spec
        config: Configuration dict

    Returns:
        AOT metadata if cached, None otherwise
    """
    cache_key = _tactic_manager._make_cache_key(spec, config)
    return _tactic_manager.cache.get(cache_key)


def extract_aot_metadata_dict(metadata: AOTMetadata) -> Dict[str, Any]:
    """
    Extract AOT metadata into dict suitable for plugin field encoding.

    All backends (Triton, CuTile, CuTeDSL) expose a unified interface:
    .binary (bytes), .kernel_name (str), .launch_params, .backend (str).
    """
    if not hasattr(metadata, "binary") or not hasattr(metadata, "kernel_name"):
        raise TTAPluginError(
            f"AOT metadata of type '{type(metadata).__name__}' must expose .binary and .kernel_name attributes",
            stage="compile",
            backend="unknown",
        )
    if not hasattr(metadata, "launch_params") or metadata.launch_params is None:
        raise TTAPluginError(
            f"AOTMetadata for kernel '{metadata.kernel_name}' is missing 'launch_params'. "
            "The backend may not have completed AOT compilation successfully.",
            stage="compile",
            backend=getattr(metadata, "backend", "unknown"),
        )
    required_fields = ("grid", "block", "shared_mem", "param_binding_indices", "sym_int_exprs")
    missing = [f for f in required_fields if not hasattr(metadata.launch_params, f)]
    if missing:
        raise TTAPluginError(
            f"launch_params for kernel '{metadata.kernel_name}' is missing fields: {missing}. "
            "The AOT backend must populate all launch_params fields.",
            stage="compile",
            backend=getattr(metadata, "backend", "unknown"),
        )
    launch_params = metadata.launch_params
    binary_data = (
        metadata.binary
        if isinstance(metadata.binary, bytes)
        else metadata.binary.encode("utf-8")
    )
    kernel_name = metadata.kernel_name
    backend = getattr(metadata, "backend", None)
    if backend is None:
        raise TTAPluginError(
            f"AOT metadata of type '{type(metadata).__name__}' must expose a .backend attribute (one of 'triton', 'cutile', 'cutedsl')",
            stage="compile",
            backend="unknown",
        )
    return {
        "backend": backend,
        "kernel_name": kernel_name,
        "binary_data": binary_data,
        "ptx_bytes": binary_data,  # Alias for compatibility
        "grid": launch_params.grid,
        "block": launch_params.block,
        "shared_mem": launch_params.shared_mem,
        "param_binding_indices": launch_params.param_binding_indices,
        "sym_int_exprs": launch_params.sym_int_exprs,
    }
