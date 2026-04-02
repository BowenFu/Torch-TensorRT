"""FX pass that lowers ``tta.lower_as`` annotated regions to their target impls.

For each region recorded in the exported program's ``_tta.region_table``,
this pass resolves the implementation, verifies IO compatibility, and rewrites
the region's subgraph to a single ``call_function`` node targeting the
resolved op.  Regions whose impl cannot be resolved are either skipped with a
warning (``require=False``) or raise ``LowerAsError`` (``require=True``).

Region scope is determined from the ``RegionView`` stored in
``ep._tta["region_views"]``, which is the sole authoritative post-capture
representation of region membership.  ``node.meta["tta_regions"]`` is a
transient capture-time tag that is deleted after ``build_annotation_ir``; this
pass never reads it.
"""

from __future__ import annotations

import logging
import operator as _op
from typing import Any, Dict, FrozenSet, List, Set

import torch._ops as _torch_ops
from torch._ops import OpOverload as _TorchOpOverload
from torch.fx import GraphModule, Node

from ._errors import LowerAsError
from .ir.region_discovery import BoundaryTensor
from .ir.region_discovery import RegionIO as BoundaryRegionIO
from .ir.region_view import derive_region_nodes_from_region_view
from .ir.types import RegionView
from .ir.boundary_validation import io_compatible
from ._registry import get_or_create_op_for_boundary

logger = logging.getLogger(__name__)


def _rv_to_boundary_io(rv: RegionView) -> BoundaryRegionIO:
    """Build a ``BoundaryRegionIO`` from a ``RegionView`` for ``io_compatible``.

    ``io_compatible`` expects ``BoundaryTensor`` objects that carry tensor
    metadata for dtype/rank checks.  The RegionView stores the same boundary
    edges as Endpoints (node + flattened index); this helper reads tensor
    metadata from the endpoint nodes' ``node.meta`` so the existing
    ``io_compatible`` function can be reused unchanged.

    Input boundary: ``rio.outside.node`` is the outside producer.
    Output boundary: ``rio.inside.node`` is the inside producer that escapes.
    """
    # Deduplicate by outside.node: RegionViews record one edge per boundary
    # crossing, so the same outside node may appear multiple times when a
    # single tensor crosses into the region at several entry points (e.g.
    # RMSNorm where `x` is used in both `pow` and `div`).  io_compatible and
    # get_or_create_op_for_boundary must see unique inputs, matching the
    # deduplicated count that the original compute_region_io returned.
    _seen: Set[Any] = set()
    _dedup_inputs: List[Any] = []
    for rio in rv.inputs:
        if rio.outside.node not in _seen:
            _seen.add(rio.outside.node)
            _dedup_inputs.append(rio)
    inputs = [
        BoundaryTensor(
            node=rio.outside.node,
            tensor_meta=(
                rio.outside.node.meta.get("tensor_meta")
                or rio.outside.node.meta.get("val")
            ),
        )
        for rio in _dedup_inputs
    ]
    outputs = [
        BoundaryTensor(
            node=rio.inside.node,
            tensor_meta=(
                rio.inside.node.meta.get("tensor_meta")
                or rio.inside.node.meta.get("val")
            ),
        )
        for rio in rv.outputs
    ]
    return BoundaryRegionIO(inputs=inputs, outputs=outputs)


def _rewrite_region_to_target_call(
    gm: GraphModule,
    target: Any,
    rv: RegionView,
    region_nodes: List[Node],
    *,
    node_name: str = "",
) -> FrozenSet[Node]:
    """Replace the region subgraph with a single ``call_function`` to *target*.

    Inserts one new node (or one node plus ``getitem`` unpacking for multi-output
    regions) before the first region node, redirects all downstream users, then
    erases the original region nodes.

    Args:
        gm:           The :class:`GraphModule` whose graph is being mutated.
        target:       The callable target for the replacement ``call_function`` node.
        rv:           RegionView whose ``inputs[*].outside.node`` are the input
                      producers and ``outputs[*].inside.node`` are the inside
                      producers that must be replaced.
        region_nodes: Ordered list of FX nodes belonging to the region (will be
                      erased after the rewrite).  Must be in topological order.
        node_name:    Optional name for the inserted FX node.  When provided the
                      node's ``.name`` is set to this value so that TRT converters
                      receive it as the layer name (and thus ``torch_op`` in TTA
                      layer metadata) without a global side-table lookup.

    Returns:
        A frozen set containing every newly inserted node.
    """
    graph = gm.graph
    first_region_node = region_nodes[0]
    # Build input_nodes from rv.inputs, which preserves duplicate outside
    # references (e.g. the same tensor used at two different region entry
    # points, as in RMSNorm where `x` crosses the boundary twice).
    #
    # For chained regions a previous rewrite may have erased an outside node
    # and replaced all its uses (via replace_all_uses_with).  Detect this and
    # resolve the stale reference by reading the current producer from the
    # inside endpoint's current args at the recorded flattened index.
    # Build deduplicated input_nodes from rv.inputs.  Two rules:
    # 1. Stale outside nodes (erased by a prior chained rewrite) are resolved
    #    by reading the current arg at inside.idx of the inside endpoint.
    # 2. Deduplicate by resolved node identity — the same outside tensor may
    #    cross the boundary multiple times (e.g. RMSNorm `x` enters at both
    #    `pow` and `div`), but the custom plugin receives it only once.
    _live_graph_nodes: Set[Node] = set(graph.nodes)
    _seen_inputs: Set[Node] = set()
    input_nodes: List[Node] = []
    for rio in rv.inputs:
        _outside: Any = rio.outside.node
        if _outside not in _live_graph_nodes:
            # Stale: resolve via inside.idx
            _inside = rio.inside.node
            _flat: List[Node] = []
            def _collect(_v: Any) -> None:  # noqa: E306
                if isinstance(_v, Node):
                    _flat.append(_v)
                elif isinstance(_v, (tuple, list)):
                    for _x in _v:
                        _collect(_x)
            _collect(_inside.args)
            _idx = rio.inside.idx
            _outside = _flat[_idx] if _idx < len(_flat) else None
        if _outside is not None and _outside not in _seen_inputs:
            _seen_inputs.add(_outside)
            input_nodes.append(_outside)

    with graph.inserting_before(first_region_node):
        impl_node = graph.call_function(
            target,
            args=tuple(input_nodes),
            kwargs={},
            name=node_name if node_name else None,
        )
    if node_name:
        impl_node.meta["layer_metadata"] = node_name

    # Propagate fake-tensor metadata from the first input.  When the rewrite
    # runs post-decomposition (autotune lowering pass) every existing node has
    # meta["val"]; newly inserted nodes do not.  Downstream compile stages
    # (e.g. get_output_dtypes) require this metadata to determine output dtype.
    # For pre-decomp rewrites (tta.lower_as) the metadata is not yet present
    # on input nodes either, so the check is safe either way.
    if input_nodes and impl_node.meta.get("val") is None:
        _src_val = input_nodes[0].meta.get("val")
        if _src_val is not None:
            impl_node.meta["val"] = _src_val

    new_nodes: Set[Node] = {impl_node}
    if len(rv.outputs) == 1:
        rv.outputs[0].inside.node.replace_all_uses_with(impl_node)
    else:
        with graph.inserting_after(impl_node):
            for idx, rio in enumerate(rv.outputs):
                item = graph.call_function(_op.getitem, args=(impl_node, idx))
                rio.inside.node.replace_all_uses_with(item)
                new_nodes.add(item)

    for n in reversed(region_nodes):
        if not n.users:
            graph.erase_node(n)

    graph.lint()
    gm.recompile()
    return frozenset(new_nodes)


def apply_lower_as_regions(
    exported_program: Any,
    fx_module: GraphModule,
    **_: Any,
) -> None:
    """Apply the lower_as lowering pass to all annotated regions in *fx_module*.

    Iterates over every ``"lower_as"`` entry in the exported program's
    ``_tta.region_table``.  For each entry the pass:

    1. Looks up the ``RegionView`` from ``ep._tta["region_views"]`` — the
       authoritative post-capture scope representation.  Node-meta
       ``tta_regions`` tags are transient and are not read here.
    2. Derives the set of live region nodes via
       :func:`derive_region_nodes_from_region_view`.  An empty live-node set
       means the region was already rewritten in an earlier pass; the entry is
       silently skipped with a debug log.
    3. Resolves the ``impl_id`` against ``impl_registry``.
    4. Verifies IO compatibility between the RegionView boundary and the impl
       descriptor.
    5. Materialises a TTA leaf op via :func:`get_or_create_op_for_boundary`.
    6. Rewrites the region subgraph to a single ``call_function`` node.

    When *require* is ``True`` for a region any failure in steps 3–5 raises
    :class:`LowerAsError` instead of emitting a warning.

    Args:
        exported_program: The ``ExportedProgram`` (or compatible object) whose
                          ``_tta`` attribute holds ``region_table``,
                          ``impl_registry``, and ``region_views``.
        fx_module:        The :class:`GraphModule` to rewrite in place.
        **_:              Extra keyword arguments are accepted and ignored for
                          forward-compatibility.
    """
    tta_meta: Dict[str, Any] = getattr(exported_program, "_tta", None) or {}
    region_table: Dict[int, Any] = tta_meta.get("region_table", {})
    impl_registry: Dict[str, Any] = tta_meta.get("impl_registry", {})
    region_views: Dict[int, RegionView] = tta_meta.get("region_views", {})

    # Build a live-node set once; used to filter stale RegionView references
    # (nodes erased by an earlier rewrite pass or by run_decompositions()).
    live_nodes: Set[Node] = set(fx_module.graph.nodes)

    for region_id, rec in list(region_table.items()):
        if rec.get("kind") != "lower_as":
            continue

        args: Dict[str, Any] = rec.get("args", {}) or {}
        impl_id: str = args.get("impl_id", "")
        constraints: Dict[str, Any] = rec.get("constraints", {}) or {}
        require: bool = bool(constraints.get("require", False))
        name: str = rec.get("name") or f"lower_as_region_{region_id}"

        # --- Step 0: look up RegionView ----------------------------------------
        rv: RegionView | None = region_views.get(region_id)
        if rv is None:
            logger.warning(
                "tta.lower_as '%s' (rid=%s, impl_id='%s'): no RegionView found in "
                "ep._tta['region_views'].  The region may not have been registered "
                "before build_annotation_ir() ran.  Skipping.",
                name,
                region_id,
                impl_id,
            )
            continue

        # --- Step 1: derive live region nodes from RegionView ------------------
        # derive_region_nodes_from_region_view does a DFS from input-inside
        # endpoints.  After an earlier rewrite those endpoints are erased from
        # the graph; intersecting with live_nodes yields an empty set, which we
        # use as the "already rewritten" signal.
        _candidate_nodes = derive_region_nodes_from_region_view(fx_module, rv)
        live_region_nodes = _candidate_nodes.intersection(live_nodes)
        if not live_region_nodes:
            logger.debug(
                "tta.lower_as '%s' (rid=%s): no live nodes found via RegionView "
                "(region was already rewritten in an earlier pass).",
                name,
                region_id,
            )
            continue

        # Sort region nodes in topological (graph) order so that the first node
        # in the list is the correct insertion point and erasure order is safe.
        region_nodes: List[Node] = [
            n for n in fx_module.graph.nodes if n in live_region_nodes
        ]

        # --- Step 2: resolve impl_id -------------------------------------------
        if impl_id not in impl_registry:
            msg = (
                f"tta.lower_as '{name}' (rid={region_id}): cannot resolve "
                f"impl_id='{impl_id}'. "
                f"Available impl_ids: {sorted(impl_registry.keys()) or '(none)'}."
            )
            if require:
                raise LowerAsError(
                    msg,
                    rid=region_id,
                    name=name,
                    reason="IMPL_NOT_FOUND",
                )
            logger.warning(msg)
            continue

        impl_spec: Any = impl_registry[impl_id]

        # --- Step 3: IO compatibility ------------------------------------------
        boundary_io = _rv_to_boundary_io(rv)
        ok: bool
        reason: str
        ok, reason = io_compatible(boundary_io, impl_spec)
        if not ok:
            n_in = len(rv.inputs)
            n_out = len(rv.outputs)
            msg = (
                f"tta.lower_as '{name}' (rid={region_id}, impl_id='{impl_id}'): "
                f"IO incompatible — {reason}. "
                f"Region boundary has {n_in} input(s) and {n_out} output(s). "
                f"Check that the impl descriptor declares matching arity and "
                f"compatible dtypes/ranks."
            )
            if require:
                raise LowerAsError(
                    msg,
                    rid=region_id,
                    name=name,
                    reason="IO_MISMATCH",
                )
            logger.warning(msg)
            continue

        # --- Step 4: materialise op --------------------------------------------
        # Use the unique input count (same as boundary_io.inputs after
        # deduplication in _rv_to_boundary_io) to match the plugin arity.
        try:
            op_target = get_or_create_op_for_boundary(
                impl_spec, len(boundary_io.inputs)
            )
        except (TypeError, RuntimeError) as e:
            msg = (
                f"tta.lower_as '{name}' (rid={region_id}, impl_id='{impl_id}'): "
                f"failed to materialise leaf op — {e}. "
                f"Ensure the impl spec is a recognised type "
                f"(RegistryPluginSpec, BuiltinSpec, or CustomPluginSpec)."
            )
            if require:
                raise LowerAsError(
                    msg,
                    rid=region_id,
                    name=name,
                    reason="OP_MATERIALISE_FAILED",
                ) from e
            logger.warning(msg)
            continue

        _rewrite_region_to_target_call(
            fx_module, op_target, rv, region_nodes, node_name=name
        )
        # Persist op_target → name in ep._tta so apply_pre_tagging can restore
        # layer_metadata after run_decompositions() creates fresh nodes without
        # node.meta.  ep._tta is the only structure that survives decompositions.
        _tta = getattr(exported_program, "_tta", None) or {}
        _op_name_map: Dict[str, str] = dict(_tta.get("lower_as_op_to_name") or {})
        _op_name_map[str(op_target)] = name
        _tta["lower_as_op_to_name"] = _op_name_map
        exported_program._tta = _tta
