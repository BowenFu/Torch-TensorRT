"""tta.lower_as: region-scoped annotation and per-export config."""

from __future__ import annotations

from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import _capture_state as cs
from ._custom_plugin._descriptor import CustomPluginSpec
from ._specs import BuiltinSpec, KernelImplSpec, RegistryPluginSpec


def spec_to_impl_id(impl: Any) -> str:
    """Return a stable string identifier for *impl*, used as keys in region_table and impl_registry.

    The ID is deterministic for spec types that declare ``to_cache_key()``
    (i.e. :class:`BuiltinSpec`, :class:`RegistryPluginSpec`)
    and is the ``op_name`` for :class:`CustomPluginSpec`.  For any other
    object the fallback is ``str(id(impl))``, which is unique within a process
    but not stable across restarts.

    Args:
        impl: An implementation spec or descriptor object.

    Returns:
        A non-empty string that uniquely identifies *impl* within the current
        process for the purposes of region_table look-ups.
    """
    if isinstance(impl, (BuiltinSpec, RegistryPluginSpec)):
        return str(impl.to_cache_key())
    if isinstance(impl, CustomPluginSpec):
        return impl.op_name
    if isinstance(impl, KernelImplSpec):
        return str(impl.to_cache_key())
    return str(id(impl))


@dataclass
class LowerAsRegionConfig:
    """Configuration for a single lower_as region, stored in capture-time state.

    Fields:
      impl:      The implementation spec (BuiltinSpec, RegistryPluginSpec, CustomPluginSpec, etc.).
      require:   Whether to raise LowerAsError if the region cannot be lowered.
      name:      Optional human-readable name used in diagnostics.
      region_id: Integer region ID assigned by _capture_state.next_region_id().
    """

    impl: Any
    require: bool
    name: Optional[str]
    region_id: int


def record_lower_as_region(cfg: LowerAsRegionConfig) -> None:
    """Record a lower_as region configuration into the capture-time state.

    Delegates to :func:`_capture_state.record_lower_as_region_entry` using the
    region ID from *cfg* as the key.

    Args:
        cfg: The :class:`LowerAsRegionConfig` to store.
    """
    cs.record_lower_as_region_entry(cfg.region_id, cfg)


def get_all_lower_as_regions() -> Dict[int, LowerAsRegionConfig]:
    """Return all currently recorded lower_as region configurations.

    Returns:
        A mapping from region ID (int) to :class:`LowerAsRegionConfig`, as
        accumulated during the current capture session.  Returns an empty dict
        when no regions have been recorded or capture mode is not active.
    """
    return cs.get_lower_as_region_entries()  # type: ignore[return-value]


def clear_lower_as_regions() -> None:
    """Clear all lower_as region entries from the capture-time state.

    Intended for use between export calls (e.g. in tests) to reset any
    previously recorded regions so that they do not bleed into the next capture.
    """
    cs.clear_lower_as_region_entries()


# Alias for naming consistency with reset_region_table() in _capture_state.
reset_lower_as_regions = clear_lower_as_regions


class lower_as(ContextDecorator):
    """Region-scoped annotation for mapping an inline region to a single Impl.

    Usage::

        with tta.lower_as(impl=..., require=..., name=...):
            # arbitrary PyTorch code

    **Eager mode**: no effect — the context manager is a transparent no-op.

    **Export / capture mode**: allocates a ``region_id``, records a
    :class:`LowerAsRegionConfig`, and pushes/pops the region ID on the active
    region stack so that FX nodes created inside the block are tagged with
    ``tta_regions=[region_id]`` in their ``node.meta``.

    Because this class inherits :class:`~contextlib.ContextDecorator` it can
    also be used as a function decorator::

        @tta.lower_as(impl=my_plugin_spec)
        def forward(x):
            ...

    Args:
        impl:    The implementation spec that the region should be lowered to
                 (e.g. a :class:`RegistryPluginSpec` or :class:`BuiltinSpec`).
        require: If ``True``, the lowering pass will raise
                 :class:`~._errors.LowerAsError` when the region cannot
                 be lowered instead of emitting a warning and skipping it.
                 Defaults to ``False``.
        name:    Optional human-readable label used in diagnostics and log
                 messages.  Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        impl: Any,
        require: bool = False,
        name: Optional[str] = None,
    ) -> None:
        self._impl = impl
        self._require = bool(require)
        self._name = name
        self._region_id: Optional[int] = None

    def __enter__(self) -> lower_as:
        # NOTE: __enter__ returns ``self`` (a ``lower_as`` instance) rather than
        # ``None``.  This is intentional: ``lower_as`` inherits ``ContextDecorator``
        # so it can also be used as a function decorator (@tta.lower_as(impl=...)),
        # which requires __enter__ to return self.  The sibling context managers
        # (tta.quantize, tta.autocast, tta.autotune) are plain @contextmanager
        # generators that yield None and therefore cannot be used as decorators.
        cs._ensure_state()
        if not cs.in_capture_mode():
            return self
        region_id = cs.next_region_id()
        self._region_id = region_id
        cfg = LowerAsRegionConfig(
            impl=self._impl,
            require=self._require,
            name=self._name,
            region_id=region_id,
        )
        record_lower_as_region(cfg)
        cs.push_region(region_id)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc: Optional[BaseException],
        tb: Any,
    ) -> bool:
        cs._ensure_state()
        if cs.in_capture_mode() and self._region_id is not None:
            cs.pop_region(self._region_id)
        return False
