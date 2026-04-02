"""TTA capture state: region stack and node tagging during export.

Used by lower_as, quantize, autocast, autotune, and other region-scoped annotations.
When capture mode is enabled, FX nodes created during torch.export get
tta_seq and tta_regions in node.meta.

Region table
------------
``_tls.region_table`` maps region_id (int) -> RegionRecord dict:

  {
    "kind":        str,        # "quantize", "autocast", "lower_as", "autotune", ...
    "name":        str | None,
    "args":        dict,       # kind-specific args (formerly "config")
    "seq_begin":   int,
    "seq_end":     int | None,
  }

Use ``enter_region`` / ``exit_region`` to populate the table.  Use
``get_region_table`` / ``reset_region_table`` to read / clear it.

Impl registry
-------------
``_tls.impl_registry`` maps impl_id (int) -> impl descriptor object.
Used by tta.autotune to register candidate implementations at export time.
Access via ``register_impl`` / ``get_impl_registry`` / ``reset_impl_registry``.

Internal API summary
--------------------
High-level (perform nesting validation):
  ``enter_region()`` / ``exit_region()``
      Region push/pop with autotune nesting validation (R3 rule).
      Use these from user-facing context managers (quantize, autocast, autotune).

Low-level (bypass nesting validation):
  ``push_region()`` / ``pop_region()``
      Directly manipulate the active-region stack without validation.
      Used intentionally by ``lower_as``, which manages its own region lifecycle
      and must not trigger the autotune nesting check.

Process-level graph patch (lock-protected):
  ``install_graph_tagging()`` / ``uninstall_graph_tagging()``
      Monkey-patch ``torch.fx.Graph.create_node`` to tag TTA metadata onto
      every new FX node.  Protected by ``_graph_tagging_lock`` and a refcount
      so concurrent compilations share the patch safely.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from torch.fx import Graph, Node

_logger = logging.getLogger(__name__)

_tls = threading.local()

# ---------------------------------------------------------------------------
# CaptureMode — merged from _capture_mode (was a 26-line standalone module)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CaptureMode:
    """
    Global capture mode for TTA precision annotations.

    kind:
      - "none": eager or non-TTA export (tta.autocast is metadata only)
      - "amp" : Torch-TRT export; tta.autocast drives torch.amp.autocast
    """

    kind: str = "none"


# CAPTURE_MODE is a ContextVar so that each asyncio task or thread-pool worker
# that calls tta.compile() gets its own isolated value without requiring explicit
# passing of state.  Callers outside this module that need to read or set the
# current capture kind should access it through this variable directly:
#
#   token = CAPTURE_MODE.set(CaptureMode(kind="amp"))
#   try:
#       ...
#   finally:
#       CAPTURE_MODE.reset(token)
#
# The default value (kind="none") means "no active TTA export"; tta.autocast
# annotations are recorded as metadata only and do not drive torch.amp.autocast.
CAPTURE_MODE: ContextVar[CaptureMode] = ContextVar(
    "TTA_CAPTURE_MODE", default=CaptureMode()
)

# ---------------------------------------------------------------------------
# Graph-tagging lock and refcount
# ---------------------------------------------------------------------------
# install_graph_tagging / uninstall_graph_tagging mutate Graph.create_node,
# a class-level attribute shared across the entire process.  Two concurrent
# tta.compile() calls would race without this lock+refcount pair.
_graph_tagging_lock = threading.Lock()
_graph_tagging_refcount = 0  # how many compilations currently hold the patch


def _ensure_state() -> None:
    """Lazily initialise all thread-local state fields.

    WHY this pattern exists
    -----------------------
    Thread-local storage (``threading.local()``) does not pre-populate
    attributes on new threads: a freshly spawned worker thread sees an empty
    namespace.  We cannot initialise at import time (that would only initialise
    state for the importing thread) and we cannot require callers to call an
    explicit ``init()`` before every public API — that would be fragile and
    would break any code path that enters the module through an unexpected
    entry point (e.g. a test helper calling ``get_region_table()`` directly
    without first calling ``set_capture_mode()``).

    The lazy-init guard pattern solves both problems:
    - **Thread safety**: each thread initialises its own TLS namespace the
      first time any public function in this module is called from that thread.
    - **Multi-entry-point safety**: every public function calls
      ``_ensure_state()`` unconditionally, so the TLS namespace is always
      valid regardless of which function is the first to be called.

    This function is idempotent and cheap: each ``hasattr`` guard is an
    O(1) dict lookup, so calling it multiple times in a single call-chain
    (e.g. once in ``enter_region`` and again inside ``_next_seq_marker``) is
    intentional and harmless.  Callers should NOT try to deduplicate these
    calls — the redundancy is the safety net that keeps the TLS valid even
    when entry points are called in isolation.
    """
    if not hasattr(_tls, "in_capture"):
        _tls.in_capture = False
    if not hasattr(_tls, "node_seq_counter"):
        _tls.node_seq_counter = 0
    if not hasattr(_tls, "region_id_counter"):
        _tls.region_id_counter = 0
    if not hasattr(_tls, "seq_marker_counter"):
        _tls.seq_marker_counter = 0
    if not hasattr(_tls, "active_regions"):
        _tls.active_regions = []
    if not hasattr(_tls, "region_table"):
        _tls.region_table = {}
    if not hasattr(_tls, "impl_registry"):
        _tls.impl_registry = {}
    if not hasattr(_tls, "impl_id_counter"):
        _tls.impl_id_counter = 0
    if not hasattr(_tls, "lower_as_regions"):
        _tls.lower_as_regions = {}
    if not hasattr(_tls, "expect_registry"):
        _tls.expect_registry = []
    if not hasattr(_tls, "expect_stack"):
        _tls.expect_stack = []
    if not hasattr(_tls, "expect_id_counter"):
        _tls.expect_id_counter = 0
    if not hasattr(_tls, "export_as_op_to_name"):
        _tls.export_as_op_to_name = {}


def set_capture_mode(enabled: bool) -> None:
    """Enable or disable TTA capture mode for the current thread.

    When ``enabled`` is ``True``, subsequent FX node creation events (via the
    ``Graph.create_node`` monkey-patch installed by ``install_graph_tagging()``)
    will tag each node with a ``tta_seq`` sequence number and, if any regions
    are active, a ``tta_regions`` list.

    When ``enabled`` is ``False``, capture mode is turned off and the
    active-region stack is cleared.  The region table and other per-session
    state are *not* cleared here; call ``reset_region_table()`` for a full
    teardown.

    Args:
        enabled: ``True`` to enter capture mode, ``False`` to leave it.
    """
    _ensure_state()
    _tls.in_capture = bool(enabled)
    if not enabled:
        _tls.active_regions = []


def _next_seq_marker() -> int:
    """Return the next monotonic sequence number for region begin/end markers.

    This is a private function.  It unconditionally increments the counter
    regardless of capture mode; callers are responsible for ensuring this is
    only invoked during an active capture pass (i.e. after
    ``set_capture_mode(True)`` and before ``set_capture_mode(False)``).
    """
    _ensure_state()
    v = _tls.seq_marker_counter
    _tls.seq_marker_counter += 1
    return v


def enter_region(
    *,
    kind: str,
    name: Optional[str],
    args: Dict[str, Any],
) -> int:
    """Enter a TTA region and push it onto the active region stack.

    Records the region in ``_tls.region_table`` with a begin sequence marker
    and appends its id to ``_tls.active_regions``.  After this call, any FX
    nodes created (via the ``Graph.create_node`` monkey-patch) will have the
    new region_id included in their ``tta_regions`` metadata.

    Args:
        kind: Region kind string, e.g. ``"quantize"``, ``"autocast"``,
              ``"lower_as"``, or ``"autotune"``.
        name: Optional human-readable label for the region (used in error
              messages and debugging tools).
        args: Kind-specific keyword arguments dict (e.g. dtype, engine
              settings, search space).  Stored verbatim in the region table.

    Returns:
        int: The newly allocated region_id.  Pass this to ``exit_region()``
             (or ``pop_region()`` for low-level callers) when leaving the
             region.

    Raises:
        ValueError: Raised when nesting rules are violated.  Specifically,
            nesting *any* TTA region inside an ``autotune`` region whose
            ``include_baseline_in_search`` argument is ``False`` is forbidden
            (R3 in the autotune design spec).  Set
            ``include_baseline_in_search=True`` on the outer autotune region
            to allow nested annotations.
    """
    _ensure_state()
    # Nesting validation: reject any nested region inside an autotune region
    # with include_baseline_in_search=False.
    for active_rid in _tls.active_regions:
        rec = _tls.region_table.get(active_rid)
        if rec is None:
            continue
        if rec["kind"] == "autotune" and not rec["args"].get("include_baseline_in_search", True):
            raise ValueError(
                f"Cannot nest a {kind!r} region inside autotune region "
                f"rid={active_rid} (name={rec['name']!r}) which has "
                f"include_baseline_in_search=False. "
                f"Set include_baseline_in_search=True on the outer autotune "
                f"region to allow nested regions."
            )
    rid = _tls.region_id_counter
    _tls.region_id_counter += 1
    _tls.region_table[rid] = {
        "kind": kind,
        "name": name,
        "args": args,
        "seq_begin": _next_seq_marker(),
        "seq_end": None,
    }
    _tls.active_regions.append(rid)
    return rid


def exit_region(*, region_id: int) -> None:
    """Exit a TTA region: pop the active-region stack and record the end marker.

    Removes ``region_id`` from the top of ``_tls.active_regions`` and sets
    ``seq_end`` in the region table entry.  After this call, newly created FX
    nodes will no longer carry this region in their ``tta_regions`` metadata.

    Args:
        region_id: The id returned by the corresponding ``enter_region()`` call.

    Raises:
        RuntimeError: If the region stack is corrupted, i.e. ``region_id`` is
            not the top element of the active stack.  This indicates a
            lifecycle bug (mismatched enter/exit calls or exception swallowing
            inside an annotated block).
    """
    _ensure_state()
    if not _tls.active_regions or _tls.active_regions[-1] != region_id:
        raise RuntimeError(
            f"TTA region stack corrupted: expected region_id={region_id} "
            f"on top, got {_tls.active_regions[-1] if _tls.active_regions else 'empty'}"
        )
    _tls.active_regions.pop()
    if region_id in _tls.region_table:
        _tls.region_table[region_id]["seq_end"] = _next_seq_marker()


def get_region_table() -> Dict[int, Dict[str, Any]]:
    """Return a shallow copy (snapshot) of the current region table.

    The returned dict maps ``region_id`` (int) to a ``RegionRecord`` dict with
    keys ``kind``, ``name``, ``args``, ``seq_begin``, and ``seq_end``.  Mutating
    the returned dict does not affect the live state.  ``seq_end`` is ``None``
    for regions that have been entered but not yet exited.

    Call this *after* ``torch.export.export()`` completes to get the fully
    populated table (all ``seq_end`` values set).
    """
    _ensure_state()
    return dict(_tls.region_table)


def reset_region_table() -> None:
    """Clear all capture-time state: region table, lower_as regions, impl_registry,
    and all sequence counters.

    Call this after ``torch.export.export()`` completes (and after any
    post-export analysis) to release stale entries before the next compilation.
    ``tta.compile()`` calls this automatically at start and at teardown, so
    manual use is only needed in test fixtures or tools that bypass
    ``tta.compile()``.
    """
    _ensure_state()
    _tls.region_table = {}
    _tls.active_regions = []
    _tls.region_id_counter = 0
    _tls.seq_marker_counter = 0
    _tls.node_seq_counter = 0
    _tls.impl_registry = {}
    _tls.impl_id_counter = 0
    _tls.lower_as_regions = {}
    _tls.expect_registry = []
    _tls.expect_stack = []
    _tls.expect_id_counter = 0
    _tls.export_as_op_to_name = {}


def in_capture_mode() -> bool:
    """Return ``True`` if TTA capture mode is currently active on this thread.

    This is the primary gate used by ``tag_tta_on_new_node()`` and by any
    code that should only run during a ``torch.export.export()`` pass wrapped
    by ``tta.compile()``.  Safe to call from any thread at any time; returns
    ``False`` before ``set_capture_mode(True)`` has been called.

    Returns:
        bool: ``True`` if capture is active for the current thread,
              ``False`` otherwise.
    """
    _ensure_state()
    return getattr(_tls, "in_capture", False)


def next_region_id() -> int:
    """Allocate and return the next monotonic region_id for this thread."""
    _ensure_state()
    rid = _tls.region_id_counter
    _tls.region_id_counter += 1
    return rid


def next_node_seq() -> int:
    """Allocate and return the next monotonic node sequence number for this thread."""
    _ensure_state()
    seq = _tls.node_seq_counter
    _tls.node_seq_counter += 1
    return seq


def push_region(region_id: int) -> None:
    """Low-level push of region_id onto the active-region stack.

    Bypasses the nesting validation performed by ``enter_region()``.
    Use this intentionally (as ``lower_as`` does) when the caller manages
    its own region lifecycle and must not trigger the autotune nesting check.
    """
    _ensure_state()
    _tls.active_regions.append(region_id)


def pop_region(region_id: int) -> None:
    """Low-level pop of region_id from the active-region stack.

    Bypasses the stack-ordering check performed by ``exit_region()``.
    Pair with ``push_region()`` for callers (e.g. ``lower_as``) that bypass
    nesting validation deliberately.
    """
    _ensure_state()
    if _tls.active_regions and _tls.active_regions[-1] == region_id:
        _tls.active_regions.pop()
    else:
        _logger.debug(
            "pop_region: region_id=%d not on top of the active stack (stack=%s); "
            "no-op.  This may indicate a lifecycle mismatch in the caller.",
            region_id,
            _tls.active_regions,
        )


def active_regions() -> List[int]:
    """Return a snapshot of the currently active region_id stack.

    The list is ordered from outermost to innermost (insertion order); the
    last element is the most-recently entered region.  The list is unsorted —
    callers that need a canonical ordering (e.g. ``_expect/_capture.py``) must
    sort independently.
    """
    _ensure_state()
    return list(getattr(_tls, "active_regions", []))


def register_impl(impl: Any) -> int:
    """Register an impl descriptor in the per-session impl_registry.

    Assigns a monotonically increasing integer id and stores ``impl`` under
    that key.  Returns the assigned id so the caller can refer to the
    descriptor later via ``get_impl_registry()``.  The registry is cleared by
    ``reset_region_table()`` and ``reset_impl_registry()``.
    """
    _ensure_state()
    impl_id = _tls.impl_id_counter
    _tls.impl_id_counter += 1
    _tls.impl_registry[impl_id] = impl
    return impl_id


def get_impl_registry() -> Dict[int, Any]:
    """Return a shallow copy (snapshot) of the current impl_registry.

    Maps ``impl_id`` (int) to the impl descriptor registered via
    ``register_impl()``.  Mutating the returned dict does not affect live
    state.  Typically consumed by ``tta.compile()`` after export to attach
    descriptors to the exported program's ``_tta`` attribute.
    """
    _ensure_state()
    return dict(_tls.impl_registry)


def record_lower_as_region_entry(region_id: int, cfg: Any) -> None:
    """Store a lower_as region config keyed by region_id.

    Called by the ``lower_as`` context manager when it enters a region, so
    that the post-export pass can look up the engine config (device, dtype,
    compilation settings, etc.) for each tagged subgraph.

    Args:
        region_id: The region id assigned to this lower_as region.
        cfg: An opaque config object (typically a ``LowerAsConfig`` dataclass)
             describing how the subgraph should be compiled.
    """
    _ensure_state()
    _tls.lower_as_regions[region_id] = cfg


def get_lower_as_region_entries() -> Dict[int, Any]:
    """Return a shallow copy of the lower_as region table.

    Maps ``region_id`` (int) to the config object stored by
    ``record_lower_as_region_entry()``.  Mutating the returned dict does not
    affect live state.  Consumed by the post-export lowering pass.

    Returns:
        Dict[int, Any]: Snapshot of the per-region lower_as configs.
    """
    _ensure_state()
    return dict(_tls.lower_as_regions)


def clear_lower_as_region_entries() -> None:
    """Clear the lower_as region table.

    Called by ``reset_region_table()`` automatically; exposed separately for
    callers that need to clear only the lower_as state.
    """
    _ensure_state()
    _tls.lower_as_regions = {}


def record_export_as_op_name(op_key: str, name: str) -> None:
    """Store an ``op_key → name`` entry for an ``export_as`` boundary op.

    Called during the ``export_as`` tracing wrapper when capture mode is
    active.  The mapping survives until ``reset_region_table()`` or
    ``clear_export_as_op_names()`` is called.

    Args:
        op_key: String representation of the ``OpOverload``, e.g.
                ``"torch_tensorrt_anno_builtin.builtin_add_activation_abc.default"``.
        name:   User-supplied annotation name, e.g. ``"relu_block_export"``.
    """
    _ensure_state()
    _tls.export_as_op_to_name[op_key] = name


def get_export_as_op_names() -> Dict[str, str]:
    """Return a snapshot of the ``op_key → name`` map for ``export_as`` ops."""
    _ensure_state()
    return dict(_tls.export_as_op_to_name)


def clear_export_as_op_names() -> None:
    """Clear the export_as op → name mapping."""
    _ensure_state()
    _tls.export_as_op_to_name = {}


def get_expect_stack() -> List[int]:
    """Return the mutable per-thread tta.expect active region_id stack.

    The returned list is the *live* stack object, not a copy — callers may
    append/pop directly.  The ``_expect/_capture.py`` module uses this to push
    region ids when entering a ``tta.expect`` block and pop them on exit.

    Returns:
        List[int]: The live active-region stack for tta.expect regions.
    """
    _ensure_state()
    return _tls.expect_stack


def get_expect_registry() -> List[Any]:
    """Return the mutable per-thread tta.expect RegionSpec list.

    The returned list is the *live* registry object, not a copy — callers
    append ``RegionSpec`` entries as each ``tta.expect`` block is entered.
    Consumed by the evaluate pass after export.

    Returns:
        List[Any]: The live list of registered tta.expect region specs.
    """
    _ensure_state()
    return _tls.expect_registry


def next_expect_id() -> int:
    """Allocate a monotonic id for a new tta.expect region (per-thread).

    The counter starts at 0 after initialisation or ``clear_expect_registry()``
    and increments by 1 on each call.  The returned id is used as the key when
    storing ``RegionSpec`` entries.

    Returns:
        int: The next available tta.expect region id.
    """
    _ensure_state()
    _tls.expect_id_counter += 1
    return _tls.expect_id_counter


def clear_expect_registry() -> None:
    """Reset the tta.expect registry, stack, and id counter.

    Called by ``reset_region_table()`` automatically; exposed separately for
    callers that need to clear only the expect state.
    """
    _ensure_state()
    _tls.expect_registry = []
    _tls.expect_stack = []
    _tls.expect_id_counter = 0


def reset_impl_registry() -> None:
    """Clear the impl_registry and reset the impl_id counter to zero.

    Use this when you need to discard only the impl registry without touching
    the region table.  For a full teardown (after export), prefer
    ``reset_region_table()``, which resets both.
    """
    _ensure_state()
    _tls.impl_registry = {}
    _tls.impl_id_counter = 0


def tag_tta_on_new_node(node: Node) -> None:
    """Attach tta_seq (and optionally tta_regions) to a newly created FX node.

    Called from the ``Graph.create_node`` monkey-patch; no-op outside capture mode.
    """
    if not in_capture_mode():
        return
    if "tta_seq" in node.meta:
        _logger.debug(
            "tag_tta_on_new_node: overwriting existing tta_seq=%r on node %r "
            "with a new sequence number.  This is unexpected outside of test "
            "fixtures that reuse node objects.",
            node.meta["tta_seq"],
            node,
        )
    node.meta["tta_seq"] = next_node_seq()
    active = active_regions()
    if active:
        node.meta["tta_regions"] = list(active)


_orig_create_node = Graph.create_node


def _create_node_with_tta_tagging(
    self: Graph,
    op: str,
    target: object,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    name: str | None = None,
    type_expr: object = None,
) -> Node:
    node = _orig_create_node(self, op, target, args, kwargs, name, type_expr)
    tag_tta_on_new_node(node)
    return node


def install_graph_tagging() -> None:
    """Install the TTA graph-tagging monkey-patch on ``torch.fx.Graph.create_node``.

    Thread-safe via ``_graph_tagging_lock`` and a refcount: the patch is applied
    only on the first call (when the refcount transitions from 0 to 1), so
    concurrent compilations share it without double-patching.
    """
    global _graph_tagging_refcount
    with _graph_tagging_lock:
        _graph_tagging_refcount += 1
        if _graph_tagging_refcount == 1:
            Graph.create_node = _create_node_with_tta_tagging


def uninstall_graph_tagging() -> None:
    """Remove the TTA graph-tagging monkey-patch from ``torch.fx.Graph.create_node``.

    Thread-safe via ``_graph_tagging_lock`` and a refcount: the original method is
    restored only when the refcount drops to 0 (i.e. all concurrent compilations
    have finished), preventing premature removal.
    """
    global _graph_tagging_refcount
    with _graph_tagging_lock:
        _graph_tagging_refcount -= 1
        if _graph_tagging_refcount == 0:
            Graph.create_node = _orig_create_node
