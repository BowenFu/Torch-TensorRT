"""
TTA Spec Types and Annotation Metadata
=======================================

This module defines the kernel spec type hierarchy used by the TTA annotation
layer to describe how a custom QDP plugin should be compiled and registered.

Spec hierarchy
--------------

::

    CustomPluginSpec — AOT QDP plugin descriptor built by ``tta.custom_plugin()``
      ├── TritonSpec    — kernel implemented with Triton
      ├── CuTileSpec    — kernel implemented with NVIDIA CuTile (cuda-tile)
      └── CuTeDSLSpec   — kernel implemented with NVIDIA CuTe DSL

    # Internal:
    KernelImplSpec    — lightweight wrapper converted to CustomPluginSpec by the
                        lowering pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


# ── Shared helpers ────────────────────────────────────────────────────────────


def _dict_to_stable_tuple(d: Dict[str, Any]) -> tuple:
    """Return a deterministic, hashable tuple representation of a dict by sorting keys."""
    return tuple(sorted(d.items()))


def _validate_kernel_spec_fields(spec_name: str, launch_fn: Any, configs: Any) -> None:
    """Validate the common fields shared by TritonSpec, CuTileSpec, and CuTeDSLSpec."""
    if not callable(launch_fn):
        raise TypeError(
            f"{spec_name}: launch_fn must be callable, got {type(launch_fn).__name__!r}"
        )
    if configs is not None and not isinstance(configs, list):
        raise TypeError(
            f"{spec_name}: configs must be a list or None, got {type(configs).__name__!r}"
        )


# ── Metadata helpers ──────────────────────────────────────────────────────────


@dataclass
class AnnotationMetadata:
    """Metadata stored on every callable decorated with ``@tta.export_as``.

    Attributes:
        impl: The implementation spec (a ``KernelImplSpec`` or ``CustomPluginSpec``).
        kwargs: Extra keyword arguments for future extensibility.
        name: Optional human-readable label for diagnostics.
        require: If ``True``, the compiler must successfully lower this op or raise.
    """

    impl: Any
    kwargs: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    require: bool = False


_ANNOTATION_METADATA_ATTR: str = "__tta_annotation_metadata__"


def attach_annotation_metadata(fn: Callable, metadata: AnnotationMetadata) -> None:
    """Attach :class:`AnnotationMetadata` to a callable."""
    setattr(fn, _ANNOTATION_METADATA_ATTR, metadata)


def get_annotation_metadata(fn: Callable) -> Optional[AnnotationMetadata]:
    """Retrieve :class:`AnnotationMetadata` from a callable, if present."""
    return getattr(fn, _ANNOTATION_METADATA_ATTR, None)


# ── Custom kernel specs (Triton / CuTile / CuTeDSL) ──────────────────────────


@dataclass(frozen=True)
class TritonSpec:
    """Specification for a custom plugin implemented with a Triton kernel.

    Attributes:
        launch_fn: The Triton kernel entry-point.
        configs: List of launch-parameter dicts for autotuning candidates.
            ``None`` means no explicit configs.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.
        kwargs: Additional Triton-specific parameters.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    input_formats: Optional[Any] = None
    output_formats: Optional[Any] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        configs_tuple = tuple(
            tuple(sorted(cfg.items())) for cfg in (self.configs or [])
        )
        in_fmts = tuple(int(f) for f in self.input_formats) if self.input_formats else ()
        out_fmts = tuple(int(f) for f in self.output_formats) if self.output_formats else ()
        return (
            "triton",
            id(self.launch_fn),
            configs_tuple,
            _dict_to_stable_tuple(self.kwargs),
            in_fmts,
            out_fmts,
        )

    def __post_init__(self) -> None:
        _validate_kernel_spec_fields("TritonSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class CuTileSpec:
    """Specification for a custom plugin implemented with an NVIDIA CuTile kernel.

    Attributes:
        launch_fn: The CuTile kernel entry-point.
        configs: List of launch-parameter dicts for autotuning candidates.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.
        kwargs: Additional CuTile-specific parameters.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    input_formats: Optional[Any] = None
    output_formats: Optional[Any] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        configs_tuple = tuple(
            tuple(sorted(cfg.items())) for cfg in (self.configs or [])
        )
        in_fmts = tuple(int(f) for f in self.input_formats) if self.input_formats else ()
        out_fmts = tuple(int(f) for f in self.output_formats) if self.output_formats else ()
        return (
            "cutile",
            id(self.launch_fn),
            configs_tuple,
            _dict_to_stable_tuple(self.kwargs),
            in_fmts,
            out_fmts,
        )

    def __post_init__(self) -> None:
        _validate_kernel_spec_fields("CuTileSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class CuTeDSLSpec:
    """Specification for a custom plugin implemented with the NVIDIA CuTe DSL.

    Attributes:
        launch_fn: The CuTe DSL kernel entry-point.
        configs: List of launch-parameter dicts for autotuning candidates.
        arch: Optional target GPU architecture string (e.g. ``"sm_80"``).
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.
        kwargs: Additional CuTe DSL-specific parameters.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    arch: Optional[str] = None
    input_formats: Optional[Any] = None
    output_formats: Optional[Any] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        configs_tuple = tuple(
            tuple(sorted(cfg.items())) for cfg in (self.configs or [])
        )
        in_fmts = tuple(int(f) for f in self.input_formats) if self.input_formats else ()
        out_fmts = tuple(int(f) for f in self.output_formats) if self.output_formats else ()
        return (
            "cutedsl",
            id(self.launch_fn),
            self.arch,
            configs_tuple,
            _dict_to_stable_tuple(self.kwargs),
            in_fmts,
            out_fmts,
        )

    def __post_init__(self) -> None:
        _validate_kernel_spec_fields("CuTeDSLSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class KernelImplSpec:
    """Internal dataclass wrapping one or more kernel specs for a custom QDP plugin.

    Attributes:
        kernel: A single kernel spec or a list of specs for multi-variant plugins.
        meta_impl: Optional callable providing output shapes and dtypes.
    """

    kernel: Union[TritonSpec, CuTileSpec, CuTeDSLSpec, List[Union[TritonSpec, CuTileSpec, CuTeDSLSpec]]]
    meta_impl: Optional[Callable] = None

    def to_cache_key(self) -> tuple:
        if isinstance(self.kernel, list):
            kernel_key: Any = tuple(k.to_cache_key() for k in self.kernel)
        else:
            kernel_key = self.kernel.to_cache_key()
        return (
            "custom_plugin",
            kernel_key,
            id(self.meta_impl) if self.meta_impl is not None else None,
        )

    def __post_init__(self) -> None:
        _valid_kernel_types = (TritonSpec, CuTileSpec, CuTeDSLSpec)
        if isinstance(self.kernel, list):
            if not self.kernel:
                raise ValueError("KernelImplSpec: 'kernel' list cannot be empty")
            for i, k in enumerate(self.kernel):
                if not isinstance(k, _valid_kernel_types):
                    raise TypeError(
                        f"KernelImplSpec: kernel[{i}] must be TritonSpec, "
                        f"CuTileSpec, or CuTeDSLSpec — got {type(k).__name__!r}"
                    )
        else:
            if not isinstance(self.kernel, _valid_kernel_types):
                raise TypeError(
                    f"KernelImplSpec: 'kernel' must be TritonSpec, CuTileSpec, "
                    f"CuTeDSLSpec, or a list of those — got {type(self.kernel).__name__!r}"
                )
        if self.meta_impl is not None and not callable(self.meta_impl):
            raise TypeError(
                f"KernelImplSpec: 'meta_impl' must be callable or None, "
                f"got {type(self.meta_impl).__name__!r}"
            )


# ── Factory functions ─────────────────────────────────────────────────────────


def triton(
    launch_fn: Callable,
    configs: Optional[List[Dict[str, Any]]] = None,
    input_formats: Optional[Any] = None,
    output_formats: Optional[Any] = None,
    **kwargs: Any,
) -> TritonSpec:
    """Create a :class:`TritonSpec` for a Triton kernel custom plugin.

    Args:
        launch_fn: Triton kernel function.
        configs: List of autotuning config dicts.  Pass ``None`` for no configs.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.

    Returns:
        A :class:`TritonSpec` instance.
    """
    return TritonSpec(
        launch_fn=launch_fn,
        configs=configs,
        input_formats=input_formats,
        output_formats=output_formats,
        kwargs=kwargs,
    )


def cutile(
    launch_fn: Callable,
    configs: Optional[List[Dict[str, Any]]] = None,
    input_formats: Optional[Any] = None,
    output_formats: Optional[Any] = None,
    **kwargs: Any,
) -> CuTileSpec:
    """Create a :class:`CuTileSpec` for a CuTile kernel custom plugin.

    Args:
        launch_fn: CuTile kernel function.
        configs: List of autotuning config dicts.  Pass ``None`` for no configs.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.

    Returns:
        A :class:`CuTileSpec` instance.
    """
    return CuTileSpec(
        launch_fn=launch_fn,
        configs=configs,
        input_formats=input_formats,
        output_formats=output_formats,
        kwargs=kwargs,
    )


def cutedsl(
    launch_fn: Callable,
    configs: Optional[List[Dict[str, Any]]] = None,
    arch: Optional[str] = None,
    input_formats: Optional[Any] = None,
    output_formats: Optional[Any] = None,
    **kwargs: Any,
) -> CuTeDSLSpec:
    """Create a :class:`CuTeDSLSpec` for a CuTe DSL kernel custom plugin.

    Args:
        launch_fn: CuTe DSL kernel function.
        configs: List of autotuning config dicts.  Pass ``None`` for no configs.
        arch: Target GPU architecture string (e.g. ``"sm_80"``).
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.

    Returns:
        A :class:`CuTeDSLSpec` instance.
    """
    return CuTeDSLSpec(
        launch_fn=launch_fn,
        configs=configs,
        arch=arch,
        input_formats=input_formats,
        output_formats=output_formats,
        kwargs=kwargs,
    )


def normalize_impl_to_spec(
    impl: Union[
        "KernelImplSpec",
        TritonSpec,
        CuTileSpec,
        CuTeDSLSpec,
    ]
) -> Union["KernelImplSpec", "CustomPluginSpec"]:
    """Coerce an ``impl`` argument into a canonical spec object.

    Kernel specs (``TritonSpec``, ``CuTileSpec``, ``CuTeDSLSpec``) are
    automatically wrapped in a ``CustomPluginSpec`` via ``tta.custom_plugin()``.

    Args:
        impl: A kernel spec or ``KernelImplSpec`` / ``CustomPluginSpec``.

    Returns:
        A canonical spec object.

    Raises:
        TypeError: If ``impl`` is not a recognised type.
    """
    if type(impl).__name__ in ("KernelImplSpec", "CustomPluginSpec"):
        return impl  # type: ignore[return-value]

    if type(impl).__name__ in ("TritonSpec", "CuTileSpec", "CuTeDSLSpec"):
        raise TypeError(
            f"normalize_impl_to_spec: bare kernel specs ({type(impl).__name__!r}) "
            f"cannot be used directly — wrap in tta.custom_plugin(kernel, meta_impl=...) "
            f"and pass the resulting CustomPluginSpec"
        )

    raise TypeError(
        f"normalize_impl_to_spec: expected a kernel spec "
        f"(TritonSpec/CuTileSpec/CuTeDSLSpec) or KernelImplSpec/CustomPluginSpec — "
        f"got {type(impl).__name__!r}"
    )


def _custom_plugin_spec(
    kernel: Union[
        TritonSpec,
        CuTileSpec,
        CuTeDSLSpec,
        List[Union[TritonSpec, CuTileSpec, CuTeDSLSpec]],
    ],
    meta_impl: Optional[Callable] = None,
) -> KernelImplSpec:
    """Internal factory: create a :class:`KernelImplSpec` wrapping one or more kernel specs."""
    return KernelImplSpec(kernel=kernel, meta_impl=meta_impl)
