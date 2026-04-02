"""
TTA Spec Types and Annotation Metadata
=======================================

This module defines the spec type hierarchy used by the TTA annotation layer to
describe how an annotated boundary op should be lowered to TensorRT at compile time.

Spec hierarchy
--------------

::

    RegistryPluginSpec          — lower via an existing TRT plugin registry entry
    BuiltinSpec         — lower via INetworkDefinition.add_*() builtin layers
    CustomPluginSpec — lower via a user-supplied kernel (Triton/CuTile/CuTeDSL)
      built by ``tta.custom_plugin()``; carries QDP registration state
      and optional weight tensors / self_attr bindings

    # Internal (not part of the public API):
    KernelImplSpec    — lightweight internal dataclass wrapping kernel specs;
      converted to CustomPluginSpec by the lowering pass
      ├── TritonSpec    — kernel implemented with Triton (JIT or AOT)
      ├── CuTileSpec    — kernel implemented with NVIDIA CuTile (cuda-tile)
      └── CuTeDSLSpec   — kernel implemented with NVIDIA CuTe DSL

Usage flow
----------
1. A user decorates a function with ``@tta.export_as(impl=<spec>)``.
2. ``export_as`` calls :func:`normalize_impl_to_spec` to coerce the ``impl``
   argument into a canonical spec object.
3. The spec is stored in an :class:`AnnotationMetadata` instance which is
   attached to the callable via :func:`attach_annotation_metadata`.
4. During compilation the ``lower_as`` pass reads the metadata via
   :func:`get_annotation_metadata` and dispatches to the appropriate
   lowering path based on the spec type.

Metadata helpers
----------------
:class:`AnnotationMetadata`  — dataclass stored on every annotated callable
:data:`_ANNOTATION_METADATA_ATTR` — private attribute name used as storage key
:func:`attach_annotation_metadata` — low-level writer (prefer ``@tta.export_as``)
:func:`get_annotation_metadata`    — low-level reader (prefer ``@tta.export_as``)

FromMethodSelf
--------------
:class:`FromMethodSelf` and its factory :func:`self_attr` are used as sentinel
default values in function signatures to defer reading an attribute of ``self``
until the plugin is registered at compile time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


# ── Shared helpers ────────────────────────────────────────────────────────────


def _dict_to_stable_tuple(d: Dict[str, Any]) -> tuple:
    """Return a deterministic, hashable tuple representation of a dict by sorting keys.

    Used by ``to_cache_key()`` methods to produce consistent cache keys regardless
    of dict insertion order.

    Args:
        d: Mapping whose keys must be strings and whose values must be hashable.

    Returns:
        Sorted ``((key, value), ...)`` tuple suitable for use inside a larger tuple
        that will be used as a dict key or set element.
    """
    return tuple(sorted(d.items()))


def _validate_kernel_spec_fields(spec_name: str, launch_fn: Any, configs: Any) -> None:
    """
    Validate the common fields shared by TritonSpec, CuTileSpec, and CuTeDSLSpec.

    Args:
        spec_name: Class name used in error messages (e.g. "TritonSpec").
        launch_fn: Value of the ``launch_fn`` field — must be callable.
        configs:   Value of the ``configs`` field — must be a ``list`` or ``None``.

    Raises:
        TypeError: If ``launch_fn`` is not callable or ``configs`` is not a list/None.
    """
    if not callable(launch_fn):
        raise TypeError(
            f"{spec_name}: launch_fn must be callable, got {type(launch_fn).__name__!r}"
        )
    if configs is not None and not isinstance(configs, list):
        raise TypeError(
            f"{spec_name}: configs must be a list or None, got {type(configs).__name__!r}"
        )


# ── Plugin and builtin specs ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RegistryPluginSpec:
    """Specification for lowering a boundary op via an existing TensorRT plugin.

    The plugin must already be registered in the TRT plugin registry at compile
    time.  The spec fields are used to look up the creator and populate the
    ``PluginFieldCollection``.

    Attributes:
        name: Plugin name exactly as registered in the TRT plugin registry.
            Must be a non-empty string.
        version: Plugin version string (e.g. ``"1"`` or ``"1.0"``).
            Must be a non-empty string.
        namespace: Plugin namespace string.  May be empty (``""``), but not
            ``None``, because TRT uses an empty string for the default namespace.
        attrs: Mapping of additional plugin field names to their values.
            Values are encoded into a ``PluginFieldCollection`` and passed to the
            plugin creator.  Supports plain Python scalars, numpy arrays, and
            :class:`FromMethodSelf` sentinels (resolved at compile time from
            ``self`` attributes of the annotated method).  Defaults to ``{}``.
    """

    name: str
    version: str
    namespace: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        """Return a deterministic, hashable key identifying this spec for op registration.

        The key is used by the TTA registry to deduplicate op registrations: two
        ``RegistryPluginSpec`` instances with identical fields produce the same key and
        map to the same registered op.

        Returns:
            A tuple ``("plugin", name, version, namespace, sorted_attrs)`` where
            ``sorted_attrs`` is a stable tuple-of-pairs representation of ``attrs``.
        """
        return (
            "plugin",
            self.name,
            self.version,
            self.namespace,
            _dict_to_stable_tuple(self.attrs),
        )

    def __post_init__(self) -> None:
        """Validate field invariants immediately after construction.

        Raises:
            ValueError: If ``name`` or ``version`` is empty, or ``namespace`` is ``None``.
        """
        if not self.name:
            raise ValueError("RegistryPluginSpec: 'name' cannot be empty")
        if not self.version:
            raise ValueError("RegistryPluginSpec: 'version' cannot be empty")
        if self.namespace is None:
            raise ValueError(
                "RegistryPluginSpec: 'namespace' cannot be None — use '' for the default namespace"
            )


@dataclass(frozen=True)
class BuiltinSpec:
    """Specification for lowering a boundary op via a TensorRT builtin layer.

    The spec names an ``INetworkDefinition.add_*()`` method and supplies the
    keyword arguments forwarded to that constructor and/or set via ``setattr``
    on the resulting layer object.

    Attributes:
        add_name: Name of the ``INetworkDefinition.add_*`` method to call
            (e.g. ``"add_convolution_nd"``, ``"add_activation"``).
            Must start with ``"add_"`` and refer to a real method on
            ``trt.INetworkDefinition``.
        kwargs: Keyword arguments forwarded to the ``add_*`` constructor and/or
            applied as ``setattr`` calls on the resulting ``ILayer`` object.
            Required parameters (as determined by the signature model) must be
            present here; optional parameters default to TRT's own defaults.
            Supports :class:`FromMethodSelf` sentinels for runtime resolution.
            Defaults to ``{}``.
        num_outputs: Number of output tensors produced by the layer.
            All ``INetworkDefinition.add_*`` methods return exactly one layer,
            and that layer exposes its outputs via ``layer.get_output(i)``.
            This field specifies how many outputs to extract.  Must be >= 1.
            Defaults to ``1``.
    """

    add_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    num_outputs: int = 1

    def to_cache_key(self) -> tuple:
        """Return a deterministic, hashable key identifying this spec for op registration.

        Different ``kwargs`` values (e.g. ``type=ActivationType.RELU`` vs
        ``type=ActivationType.SIGMOID``) produce distinct keys so that each
        unique configuration gets its own registered op.

        Returns:
            A tuple ``("builtin", add_name, sorted_kwargs)`` where ``sorted_kwargs``
            is a stable tuple-of-pairs representation of ``kwargs``.
        """
        return ("builtin", self.add_name, _dict_to_stable_tuple(self.kwargs))

    def __post_init__(self) -> None:
        """Validate field invariants immediately after construction.

        Raises:
            ValueError: If ``add_name`` is empty or ``num_outputs`` is less than 1.
        """
        if not self.add_name:
            raise ValueError("BuiltinSpec: 'add_name' cannot be empty")
        if self.num_outputs < 1:
            raise ValueError(
                f"BuiltinSpec: 'num_outputs' must be >= 1, got {self.num_outputs}"
            )


# ── Metadata helpers ──────────────────────────────────────────────────────────


@dataclass
class AnnotationMetadata:
    """Metadata stored on every callable decorated with ``@tta.export_as``.

    This dataclass is attached to the decorated function (via
    :func:`attach_annotation_metadata`) so that the compilation pipeline can
    retrieve it with :func:`get_annotation_metadata` and dispatch to the
    appropriate TRT lowering path.

    Attributes:
        impl: The implementation spec describing *how* this boundary op should
            be lowered.  One of :class:`RegistryPluginSpec`, :class:`BuiltinSpec`,
            :class:`KernelImplSpec`, or a ``CustomPluginSpec``.
        kwargs: Extra keyword arguments supplied to the ``@tta.export_as``
            decorator that are not covered by the fixed fields.  Stored for
            future extensibility.  Defaults to ``{}``.
        name: Optional human-readable label for this annotation.  Used in
            diagnostic messages, region attribution, and debug output.
            Defaults to ``None``.
        require: If ``True``, the compiler must successfully lower this
            boundary op to TensorRT; any failure (missing plugin, tensor-IO
            mismatch, etc.) raises a :class:`~._errors.TTADiagnosticError`
            immediately rather than falling back silently.  Defaults to
            ``False``.
    """

    impl: Union[RegistryPluginSpec, BuiltinSpec, "KernelImplSpec"]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    require: bool = False


# Attribute name used to store AnnotationMetadata on annotated callables.
#
# The double-underscore prefix makes this look like a name-mangled attribute,
# which minimises the chance of accidental collision with user-defined
# attributes.  User code must never read or write this attribute directly;
# use :func:`attach_annotation_metadata` / :func:`get_annotation_metadata`.
_ANNOTATION_METADATA_ATTR: str = "__tta_annotation_metadata__"


# NOTE: attach_annotation_metadata and get_annotation_metadata are low-level
# extension points for the export_as mechanism.  Advanced users who build
# custom annotation wrappers on top of TTA may need them, but typical
# application code should use the high-level @tta.export_as decorator.


def attach_annotation_metadata(fn: Callable, metadata: AnnotationMetadata) -> None:
    """Attach :class:`AnnotationMetadata` to a callable.

    Stores ``metadata`` as an attribute on ``fn`` using the private attribute
    name :data:`_ANNOTATION_METADATA_ATTR`.  This is a low-level building
    block; prefer ``@tta.export_as(impl=...)`` in application code.

    Args:
        fn: The callable to annotate.  Must support ``setattr`` (i.e. not a
            built-in function or C extension without a ``__dict__``).
        metadata: The :class:`AnnotationMetadata` instance to attach.

    Raises:
        AttributeError: If ``fn`` does not support attribute assignment.
    """
    setattr(fn, _ANNOTATION_METADATA_ATTR, metadata)


def get_annotation_metadata(fn: Callable) -> Optional[AnnotationMetadata]:
    """Retrieve :class:`AnnotationMetadata` from a callable, if present.

    Returns the metadata previously attached by :func:`attach_annotation_metadata`
    (or ``@tta.export_as``), or ``None`` if the callable has not been annotated.

    This is a low-level building block; prefer ``@tta.export_as(impl=...)``
    in application code.

    Note:
        Because ``functools.wraps`` copies the wrapped function's ``__dict__``
        to the wrapper, metadata attached to an inner function is automatically
        accessible on the outer wrapper after ``@functools.wraps`` is applied.

    Args:
        fn: The callable to inspect.

    Returns:
        The attached :class:`AnnotationMetadata`, or ``None`` if not annotated.
    """
    return getattr(fn, _ANNOTATION_METADATA_ATTR, None)


def normalize_impl_to_spec(
    impl: Union[
        RegistryPluginSpec,
        BuiltinSpec,
        "KernelImplSpec",
        "TritonSpec",
        "CuTileSpec",
        "CuTeDSLSpec",
        Dict[str, Any],
    ]
) -> Union[RegistryPluginSpec, BuiltinSpec, "KernelImplSpec", "CustomPluginSpec"]:
    """Coerce an ``impl`` argument into a canonical spec object.

    Accepts all supported descriptor forms and returns the appropriate spec
    type.  Kernel specs (``TritonSpec``, ``CuTileSpec``, ``CuTeDSLSpec``) are
    automatically wrapped in a ``CustomPluginSpec`` via
    ``tta.custom_plugin()``.  Plain dicts are parsed as either
    ``RegistryPluginSpec`` or ``BuiltinSpec`` based on which keys are present.

    Type comparisons for ``KernelImplSpec`` and ``CustomPluginSpec`` are
    done by class name rather than ``isinstance`` to avoid circular imports.

    Args:
        impl: One of:

            * :class:`RegistryPluginSpec` — returned unchanged.
            * :class:`BuiltinSpec` — returned unchanged.
            * ``KernelImplSpec`` or ``CustomPluginSpec`` — returned
              unchanged (checked by class name to avoid circular imports).
            * :class:`TritonSpec`, :class:`CuTileSpec`, or :class:`CuTeDSLSpec`
              — wrapped in a ``CustomPluginSpec`` via
              ``_custom_plugin._descriptor.custom_plugin``.
            * ``dict`` with keys ``"name"``, ``"version"``, ``"namespace"`` →
              constructed as :class:`RegistryPluginSpec`.
            * ``dict`` with key ``"add_name"`` or ``"target"`` → constructed
              as :class:`BuiltinSpec` (``"target"`` is accepted for backward
              compatibility).

    Returns:
        A canonical spec object: one of :class:`RegistryPluginSpec`,
        :class:`BuiltinSpec`, ``KernelImplSpec``, or ``CustomPluginSpec``.

    Raises:
        TypeError: If ``impl`` is not a recognised type, or if it is a dict
            that does not match any of the known key patterns.
    """
    # Plugin / builtin specs — fast path
    if isinstance(impl, (RegistryPluginSpec, BuiltinSpec)):
        return impl

    # Custom plugin specs — check by class name to avoid circular import
    if type(impl).__name__ in ("KernelImplSpec", "CustomPluginSpec"):
        return impl

    # Kernel specs — wrap in a CustomPluginSpec
    if type(impl).__name__ in ("TritonSpec", "CuTileSpec", "CuTeDSLSpec"):
        from ._custom_plugin._descriptor import custom_plugin as _cp
        return _cp(kernel=impl, meta_impl=None)

    if isinstance(impl, dict):
        # RegistryPluginSpec: requires name + version + namespace
        if "name" in impl and "version" in impl and "namespace" in impl:
            return RegistryPluginSpec(
                name=impl["name"],
                version=impl["version"],
                namespace=impl["namespace"],
                attrs=impl.get("attrs", {}),
            )
        # BuiltinSpec: keyed by add_name
        if "add_name" in impl:
            return BuiltinSpec(
                add_name=impl["add_name"],
                kwargs=impl.get("kwargs", {}),
            )
        # Backward compatibility: "target" is an alias for add_name
        if "target" in impl:
            return BuiltinSpec(
                add_name=impl["target"],
                kwargs=impl.get("kwargs", {}),
            )
        # Dict present but no recognised key pattern
        known_patterns = (
            '{"name", "version", "namespace"[, "attrs"]}  →  RegistryPluginSpec',
            '{"add_name"[, "kwargs"]}                     →  BuiltinSpec',
            '{"target"[, "kwargs"]}                       →  BuiltinSpec (legacy)',
        )
        raise TypeError(
            f"normalize_impl_to_spec: received a dict but its keys {set(impl.keys())!r} "
            f"do not match any recognised pattern.\n"
            f"Valid dict patterns:\n"
            + "\n".join(f"  {p}" for p in known_patterns)
        )

    raise TypeError(
        f"normalize_impl_to_spec: expected RegistryPluginSpec, BuiltinSpec, "
        f"CustomPluginSpec, a kernel spec "
        f"(TritonSpec/CuTileSpec/CuTeDSLSpec), or a dict — "
        f"got {type(impl).__name__!r} ({type(impl)})"
    )


class FromMethodSelf:
    """Sentinel value that defers reading a ``self`` attribute until compile time.

    Use :func:`self_attr` (the public factory) rather than constructing this
    class directly.

    **Purpose:** When annotating an instance method with ``@tta.export_as``,
    some plugin attributes may live on ``self`` (e.g. a learned weight, a
    scalar multiplier, or a shape parameter).  Because the annotation is
    defined at class-body time — before any instance exists — we cannot
    read those values immediately.  ``FromMethodSelf`` is a placeholder that
    records *which* attribute to read; the annotation lowering code resolves
    it at compile time using the actual ``self`` object of the module instance
    being compiled.

    **When to use:** Pass a ``FromMethodSelf`` instance as a value inside the
    ``attrs`` dict of a :class:`RegistryPluginSpec` or the ``kwargs`` dict of a
    :class:`BuiltinSpec`::

        class MyModule(nn.Module):
            def __init__(self, alpha: float):
                super().__init__()
                self.alpha = alpha

            @tta.export_as(impl=tta.plugin(
                "MyPlugin", "1", "my_ns",
                alpha=tta.self_attr("alpha"),   # resolved at compile time
            ))
            def forward(self, x):
                return x * self.alpha

    **Resolution:** The compilation pipeline walks the spec's ``attrs`` /
    ``kwargs``, finds every ``FromMethodSelf`` value, calls
    ``getattr(self_obj, fms.attr_name)`` on the concrete module instance, and
    substitutes the result before constructing the TRT plugin.

    Attributes:
        attr_name: Dot-separated attribute path to resolve on ``self``
            (e.g. ``"alpha"``, ``"conv.weight"``).  Nested paths are
            resolved by iterating ``getattr`` calls along the path segments.
    """

    __slots__ = ("attr_name",)

    def __init__(self, attr_name: str) -> None:
        """
        Args:
            attr_name: Dot-separated attribute path to resolve at compile time
                (e.g. ``"alpha"``, ``"sub_module.weight"``).

        Raises:
            TypeError: If ``attr_name`` is not a string.
        """
        if not isinstance(attr_name, str):
            raise TypeError(
                f"FromMethodSelf: attr_name must be a str, got {type(attr_name).__name__!r}"
            )
        self.attr_name = attr_name

    def __repr__(self) -> str:
        return f"FromMethodSelf({self.attr_name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FromMethodSelf):
            return NotImplemented
        return self.attr_name == other.attr_name

    def __hash__(self) -> int:
        return hash(self.attr_name)

    def __lt__(self, other: "FromMethodSelf") -> bool:
        """Support sorting so that ``_dict_to_stable_tuple`` works on dicts containing sentinels."""
        return self.attr_name < other.attr_name


def self_attr(attr_name: str) -> FromMethodSelf:
    """Create a :class:`FromMethodSelf` sentinel for deferred ``self`` attribute access.

    Use this inside ``tta.plugin(...)`` or ``tta.builtin(...)`` calls when a
    plugin attribute should be read from the annotated method's ``self`` at
    compile time rather than at decoration time.

    Args:
        attr_name: Dot-separated path of the attribute to read from ``self``
            at compile time (e.g. ``"alpha"``, ``"conv.weight"``).

    Returns:
        A :class:`FromMethodSelf` sentinel that the compilation pipeline will
        resolve to ``getattr(self_obj, attr_name)`` when the containing spec
        is lowered.

    Example::

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 2.0

            @tta.export_as(impl=tta.plugin(
                "ScalePlugin", "1", "my_ns",
                scale=tta.self_attr("scale"),
            ))
            def forward(self, x):
                return x * self.scale
    """
    return FromMethodSelf(attr_name)


# ── Factory functions ─────────────────────────────────────────────────────────


def plugin(name: str, version: str, namespace: str, **attrs: Any) -> RegistryPluginSpec:
    """Create a :class:`RegistryPluginSpec` for lowering via an existing TRT plugin.

    Args:
        name: Plugin name as registered in the TRT plugin registry.
        version: Plugin version string (e.g. ``"1"``).
        namespace: Plugin namespace string (use ``""`` for the default namespace).
        **attrs: Plugin field values passed to the ``PluginFieldCollection``.
            Scalar values, numpy arrays, and :class:`FromMethodSelf` sentinels
            are all accepted.

    Returns:
        A :class:`RegistryPluginSpec` instance.

    Raises:
        ValueError: If ``name`` or ``version`` is empty, or ``namespace`` is ``None``.

    Example::

        spec = tta.plugin("MyPlugin", "1", "my_ns", alpha=0.5)
    """
    return RegistryPluginSpec(name=name, version=version, namespace=namespace, attrs=attrs)


def builtin(add_name: str, **kwargs: Any) -> BuiltinSpec:
    """Create a :class:`BuiltinSpec` for lowering via a TRT ``INetworkDefinition.add_*`` method.

    Validates that ``add_name`` starts with ``"add_"``, that the method exists
    on ``trt.INetworkDefinition`` (when TensorRT is available), and that all
    required parameters for the method are present in ``kwargs``.

    Args:
        add_name: Name of the ``INetworkDefinition.add_*`` method to call,
            e.g. ``"add_convolution_nd"``, ``"add_activation"``.
        **kwargs: Parameters forwarded to the ``add_*`` constructor and/or
            applied as ``setattr`` calls on the resulting ``ILayer``.  All
            required parameters (per the :mod:`_signature` model) must be
            supplied here.

    Returns:
        A :class:`BuiltinSpec` instance.

    Raises:
        ValueError: If ``add_name`` does not start with ``"add_"``, the method
            does not exist on ``trt.INetworkDefinition``, or required parameters
            are missing from ``kwargs``.

    Example::

        spec = tta.builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3))
    """
    if not add_name.startswith("add_"):
        raise ValueError(
            f"builtin(): add_name must start with 'add_', got: {add_name!r}. "
            f"Did you mean 'add_{add_name}'?"
        )

    # Validate that the method exists on INetworkDefinition.
    # TRT is an optional dependency at spec-definition time: users on CPU-only
    # machines should still be able to define specs; validation is deferred to
    # compile time if TRT is unavailable here.
    try:
        import tensorrt as trt
        if not hasattr(trt.INetworkDefinition, add_name):
            raise ValueError(
                f"builtin(): {add_name!r} not found on trt.INetworkDefinition. "
                f"Check spelling and TensorRT version."
            )
    except ImportError:
        import warnings
        warnings.warn(
            f"TensorRT not available; cannot validate builtin spec {add_name!r} at "
            "definition time. Validation will occur at compile time.",
            UserWarning,
            stacklevel=2,
        )

    # Check that all required parameters are provided.
    from ._signature import get_signature_model

    sig = get_signature_model(add_name)
    missing_params = sig.required_params - set(kwargs.keys())
    if missing_params:
        missing_list = ", ".join(sorted(missing_params))
        raise ValueError(
            f"builtin(): Missing required parameters for {add_name!r}: {missing_list}. "
            f"These must be provided as keyword arguments to builtin()."
        )

    return BuiltinSpec(add_name=add_name, kwargs=kwargs)


# ── Custom kernel specs (Triton / CuTile / CuTeDSL) ──────────────────────────


@dataclass(frozen=True)
class TritonSpec:
    """Specification for a custom plugin implemented with a Triton kernel.

    Used as the ``kernel`` argument to ``tta.custom_plugin()``.  The kernel
    function is registered as a TRT Quick Deployable Plugin (QDP) at compile
    time.

    Attributes:
        launch_fn: The Triton kernel entry-point.  Must be callable with
            signature ``(inputs..., outputs..., **launch_params)`` where
            ``launch_params`` are drawn from the selected config dict.
        configs: List of launch-parameter dicts, one per autotuning candidate
            (e.g. ``[{"BLOCK": 128}, {"BLOCK": 256}]``).  ``None`` means no
            explicit configs; the kernel is called with no launch parameters.
        input_formats: Optional sequence of tensor layout descriptors for each
            input tensor.  Reserved for AOT compilation; pass ``None`` for JIT.
        output_formats: Optional sequence of tensor layout descriptors for each
            output tensor.  Reserved for AOT compilation; pass ``None`` for JIT.
        kwargs: Additional Triton-specific parameters passed through to the
            plugin registration machinery.  Defaults to ``{}``.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    input_formats: Optional[Any] = None
    output_formats: Optional[Any] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        """Return a deterministic, hashable key identifying this spec for op registration.

        Uses the memory address of ``launch_fn`` (via ``id()``) as a proxy for
        kernel identity; two specs sharing the same function object will share
        the same cache key.

        Returns:
            A tuple ``("triton", id(launch_fn), configs_tuple, kwargs_tuple,
            in_fmts, out_fmts)``.
        """
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
        """Validate fields on construction.

        Raises:
            TypeError: If ``launch_fn`` is not callable or ``configs`` is not a
                list or ``None``.
        """
        _validate_kernel_spec_fields("TritonSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class CuTileSpec:
    """Specification for a custom plugin implemented with an NVIDIA CuTile kernel.

    CuTile (``cuda-tile``) is NVIDIA's tile-based GPU kernel authoring
    framework.  The kernel is registered as a TRT Quick Deployable Plugin
    (QDP) at compile time.

    Attributes:
        launch_fn: The CuTile kernel entry-point.  Must be callable with
            signature ``(inputs..., outputs..., **launch_params)`` where
            ``launch_params`` are drawn from the selected config dict.
        configs: List of launch-parameter dicts for autotuning candidates.
            ``None`` means no explicit configs.
        input_formats: Optional sequence of tensor layout descriptors for each
            input tensor.  Reserved for AOT compilation; pass ``None`` for JIT.
        output_formats: Optional sequence of tensor layout descriptors for each
            output tensor.  Reserved for AOT compilation; pass ``None`` for JIT.
        kwargs: Additional CuTile-specific parameters.  Defaults to ``{}``.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    input_formats: Optional[Any] = None
    output_formats: Optional[Any] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        """Return a deterministic, hashable key identifying this spec for op registration.

        Returns:
            A tuple ``("cutile", id(launch_fn), configs_tuple, kwargs_tuple,
            in_fmts, out_fmts)``.
        """
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
        """Validate fields on construction.

        Raises:
            TypeError: If ``launch_fn`` is not callable or ``configs`` is not a
                list or ``None``.
        """
        _validate_kernel_spec_fields("CuTileSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class CuTeDSLSpec:
    """Specification for a custom plugin implemented with the NVIDIA CuTe DSL.

    CuTe DSL (``nvidia-cutlass-dsl``) is a Python-embedded DSL for writing
    high-performance CUDA kernels using CuTe tile abstractions.  The kernel
    is registered as a TRT Quick Deployable Plugin (QDP) at compile time.

    Attributes:
        launch_fn: The CuTe DSL kernel entry-point.  Must be callable with
            signature ``(inputs..., outputs..., **launch_params)`` where
            ``launch_params`` are drawn from the selected config dict.
        configs: List of launch-parameter dicts for autotuning candidates.
            ``None`` means no explicit configs.
        arch: Optional target GPU architecture string for AOT PTX compilation
            (e.g. ``"sm_80"``, ``"sm_90"``).  When ``None``, JIT compilation
            targets the current device's architecture.
        input_formats: Optional sequence of tensor layout descriptors for each
            input tensor.  Reserved for AOT compilation; pass ``None`` for JIT.
        output_formats: Optional sequence of tensor layout descriptors for each
            output tensor.  Reserved for AOT compilation; pass ``None`` for JIT.
        kwargs: Additional CuTe DSL-specific parameters.  Defaults to ``{}``.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    arch: Optional[str] = None
    input_formats: Optional[Any] = None
    output_formats: Optional[Any] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        """Return a deterministic, hashable key identifying this spec for op registration.

        Returns:
            A tuple ``("cutedsl", id(launch_fn), arch, configs_tuple,
            kwargs_tuple, in_fmts, out_fmts)``.
        """
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
        """Validate fields on construction.

        Raises:
            TypeError: If ``launch_fn`` is not callable or ``configs`` is not a
                list or ``None``.
        """
        _validate_kernel_spec_fields("CuTeDSLSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class KernelImplSpec:
    """Internal dataclass wrapping one or more kernel specs for a custom QDP plugin.

    This is the *internal* representation.  The public-facing descriptor
    returned by ``tta.custom_plugin()`` is ``CustomPluginSpec`` (defined
    in ``_custom_plugin._descriptor``), which drives QDP registration and TRT
    plugin lowering.  When the lowering pass encounters a ``KernelImplSpec``
    it converts it to a ``CustomPluginSpec`` via ``_spec_to_descriptor``.

    Attributes:
        kernel: A single kernel spec (:class:`TritonSpec`, :class:`CuTileSpec`,
            or :class:`CuTeDSLSpec`) or a list of such specs for multi-variant
            plugins (e.g. one Triton variant and one CuTile variant for
            different GPU architectures).  Lists must be non-empty and may not
            contain any non-kernel-spec items.
        meta_impl: Optional callable that provides plugin metadata such as
            output shapes and dtypes.  Used during the TRT network-definition
            phase before kernel execution.  Must be callable or ``None``.
    """

    kernel: Union[TritonSpec, CuTileSpec, CuTeDSLSpec, List[Union[TritonSpec, CuTileSpec, CuTeDSLSpec]]]
    meta_impl: Optional[Callable] = None

    def to_cache_key(self) -> tuple:
        """Return a deterministic, hashable key identifying this spec for op registration.

        Returns:
            A tuple ``("custom_plugin", kernel_key, meta_impl_id)`` where
            ``kernel_key`` is either a single ``to_cache_key()`` result or a
            tuple of them (for multi-variant plugins), and ``meta_impl_id`` is
            ``id(meta_impl)`` or ``None``.
        """
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
        """Validate fields on construction.

        Raises:
            ValueError: If ``kernel`` is an empty list.
            TypeError: If ``kernel`` or any element of a kernel list is not a
                recognised kernel spec, or if ``meta_impl`` is not callable.
        """
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


# ── Custom Plugin Factory Functions ───────────────────────────────────────────


def triton(
    launch_fn: Callable,
    configs: Optional[List[Dict[str, Any]]] = None,
    input_formats: Optional[Any] = None,
    output_formats: Optional[Any] = None,
    **kwargs: Any,
) -> TritonSpec:
    """Create a :class:`TritonSpec` for a Triton kernel custom plugin.

    Args:
        launch_fn: Triton kernel function with signature
            ``(inputs..., outputs..., **launch_params)``.
        configs: List of autotuning config dicts (e.g.
            ``[{"BLOCK": 128}, {"BLOCK": 256}]``).  Pass ``None`` for a
            kernel with no launch parameters.
        input_formats: Optional tensor layout descriptors for input tensors.
            Reserved for future AOT compilation support; pass ``None`` for JIT.
        output_formats: Optional tensor layout descriptors for output tensors.
            Reserved for future AOT compilation support; pass ``None`` for JIT.
        **kwargs: Additional Triton-specific parameters forwarded to
            :class:`TritonSpec`.

    Returns:
        A :class:`TritonSpec` instance.

    Example::

        @tta.export_as(impl=tta.custom_plugin(
            tta.triton(my_kernel, configs=[{"BLOCK": 128}])
        ))
        def my_op(x):
            return x * 2
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
        launch_fn: CuTile kernel function with signature
            ``(inputs..., outputs..., **launch_params)``.
        configs: List of autotuning config dicts.  Pass ``None`` for a kernel
            with no launch parameters.
        input_formats: Optional tensor layout descriptors for input tensors.
            Reserved for future AOT compilation support; pass ``None`` for JIT.
        output_formats: Optional tensor layout descriptors for output tensors.
            Reserved for future AOT compilation support; pass ``None`` for JIT.
        **kwargs: Additional CuTile-specific parameters forwarded to
            :class:`CuTileSpec`.

    Returns:
        A :class:`CuTileSpec` instance.

    Example::

        @tta.export_as(impl=tta.custom_plugin(
            tta.cutile(my_kernel, configs=[{"TILE_M": 64}])
        ))
        def my_op(x):
            return x + 1
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
        launch_fn: CuTe DSL kernel function with signature
            ``(inputs..., outputs..., **launch_params)``.
        configs: List of autotuning config dicts.  Pass ``None`` for a kernel
            with no launch parameters.
        arch: Target GPU architecture for AOT PTX generation
            (e.g. ``"sm_80"``, ``"sm_90"``).  ``None`` targets the current
            device at JIT compile time.
        input_formats: Optional tensor layout descriptors for input tensors.
            Reserved for future AOT compilation support; pass ``None`` for JIT.
        output_formats: Optional tensor layout descriptors for output tensors.
            Reserved for future AOT compilation support; pass ``None`` for JIT.
        **kwargs: Additional CuTe DSL-specific parameters forwarded to
            :class:`CuTeDSLSpec`.

    Returns:
        A :class:`CuTeDSLSpec` instance.

    Example::

        @tta.export_as(impl=tta.custom_plugin(
            tta.cutedsl(my_kernel, arch="sm_90", configs=[{}])
        ))
        def my_op(x):
            return x.transpose(0, 1)
    """
    return CuTeDSLSpec(
        launch_fn=launch_fn,
        configs=configs,
        arch=arch,
        input_formats=input_formats,
        output_formats=output_formats,
        kwargs=kwargs,
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
    """Create a :class:`KernelImplSpec` wrapping one or more kernel specs.

    This is an *internal* factory used by the TTA machinery.  The
    public-facing API is ``tta.custom_plugin()`` (defined in
    ``_custom_plugin._descriptor``), which returns a ``CustomPluginSpec``
    backed by QDP.  The ``__init__.py`` import of ``custom_plugin`` from
    ``_custom_plugin`` shadows any same-named symbol that might otherwise be
    exported from this module.

    Args:
        kernel: A single kernel spec or a list of kernel specs for
            multi-variant plugins.
        meta_impl: Optional callable providing plugin metadata (e.g. output
            shapes and dtypes) during TRT network definition.

    Returns:
        A :class:`KernelImplSpec` instance.
    """
    return KernelImplSpec(kernel=kernel, meta_impl=meta_impl)
