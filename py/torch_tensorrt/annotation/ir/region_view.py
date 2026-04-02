"""Post-export TTA-IR construction: RegionView, children_by_kind, etc.

All functions here operate on the *final* FX graph (after torch.export) and
on the node_regions map {node: [r0, r1, ..., rk]} captured during export.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .types import Endpoint, RegionIO, RegionView


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_fx_node(v: Any) -> bool:
    """Return True if ``v`` looks like an FX node (duck-typed)."""
    return hasattr(v, "op") and hasattr(v, "args") and hasattr(v, "users")


def _flatten_input_occurrences(consumer: Any) -> Dict[Any, List[int]]:
    """Flatten consumer.args to a list of FX nodes and record their indices.

    Traverses ``consumer.args`` depth-first, recursing into tuples and lists.
    Records all flattened positions at which each producer node appears.

    Args:
      consumer: An FX node whose args may contain nested tuples/lists of nodes.

    Returns:
      Mapping from producer_node to sorted list of flattened indices at which
      that producer appears in the flattened args sequence.
    """
    flat: List[Any] = []

    def collect(val: Any) -> None:
        if _is_fx_node(val):
            flat.append(val)
        elif isinstance(val, (tuple, list)):
            for x in val:
                collect(x)

    collect(consumer.args)

    occ: Dict[Any, List[int]] = {}
    for idx, n in enumerate(flat):
        occ.setdefault(n, []).append(idx)
    return occ


def _list_diff(a: List[int], b: List[int]) -> List[int]:
    """Return elements of ``a`` that are not in ``b``, preserving order.

    Args:
      a: The source list.
      b: Elements to exclude.

    Returns:
      Ordered list of elements in ``a`` absent from ``b``.
    """
    sb = set(b)
    return [x for x in a if x not in sb]


# ---------------------------------------------------------------------------
# RegionView construction
# ---------------------------------------------------------------------------

def _iter_all_nodes(gm: Any):
    """Yield every FX node from gm and any immediate child GraphModules.

    torch.export wraps higher-order-op bodies (e.g. wrap_with_autocast) as
    immediate child GraphModules.  Region boundaries inside those bodies are
    invisible to main-graph-only traversal, so child graphs must be included.
    """
    yield from gm.graph.nodes
    for _name, child in gm.named_children():
        if hasattr(child, "graph"):
            yield from child.graph.nodes


def build_region_views_post_export(
    gm: Any,
    *,
    node_regions: Dict[Any, List[int]],
    region_ids: List[int],
) -> Dict[int, RegionView]:
    """Build a RegionView for each region id from the final FX graph.

    For each producer→consumer edge in the graph:
      - If the consumer is in regions the producer is not: RegionView.inputs.
      - If the producer is in regions the consumer is not: RegionView.outputs.

    Args:
      gm:           FX GraphModule (post-export).
      node_regions: {node: [r0, r1, ..., rk]} outermost→innermost membership
                    stack for each node.
      region_ids:   List of region ids to build RegionViews for.

    Returns:
      Mapping {rid: RegionView} for all region ids in ``region_ids``.
    """
    views: Dict[int, RegionView] = {
        rid: RegionView(region_id=rid) for rid in region_ids
    }

    # Pre-compute per-consumer flattened input occurrences for all graphs
    # (main graph + HOp child sub-graphs such as wrap_with_autocast bodies).
    consumer_occ: Dict[Any, Dict[Any, List[int]]] = {}
    for node in _iter_all_nodes(gm):
        consumer_occ[node] = _flatten_input_occurrences(node)

    for consumer in _iter_all_nodes(gm):
        cons_stack = node_regions.get(consumer, [])
        occ = consumer_occ[consumer]

        for producer in consumer.all_input_nodes:
            prod_stack = node_regions.get(producer, [])
            if prod_stack == cons_stack:
                continue

            entered = _list_diff(cons_stack, prod_stack)   # regions entered
            exited  = _list_diff(prod_stack, cons_stack)   # regions exited

            if not entered and not exited:
                continue

            for input_idx in occ.get(producer, []):
                for rid in entered:
                    if rid in views:
                        views[rid].inputs.append(
                            RegionIO(
                                outside=Endpoint(node=producer, idx=0),
                                inside=Endpoint(node=consumer, idx=input_idx),
                            )
                        )
                for rid in exited:
                    if rid in views:
                        views[rid].outputs.append(
                            RegionIO(
                                inside=Endpoint(node=producer, idx=0),
                                outside=Endpoint(node=consumer, idx=input_idx),
                            )
                        )

    return views


# ---------------------------------------------------------------------------
# children_by_kind
# ---------------------------------------------------------------------------

def derive_region_children_by_kind(
    node_regions: Dict[Any, List[int]],
    region_table: Dict[int, Dict[str, Any]],
) -> Dict[str, Dict[int, Set[int]]]:
    """Derive same-kind parent→child region relationships from node_regions.

    ``children_by_kind[kind][rid]`` is the set of direct child region ids of
    the same kind as ``rid``.  Cross-kind parent/child pairs are ignored.

    Rules:
      For each adjacent pair (stack[i], stack[i+1]) in any node's membership
      stack: if both regions have the same kind, stack[i+1] is recorded as a
      direct child of stack[i] for that kind.

    Args:
      node_regions:  {node: [r0, r1, ..., rk]} outermost→innermost.
      region_table:  rid -> region record dict (must contain "kind" key).

    Returns:
      {kind: {parent_rid: set of child_rids}} for each kind in region_table.
    """
    kinds = {rec["kind"] for rec in region_table.values()}

    children_by_kind: Dict[str, Dict[int, Set[int]]] = {}
    for kind in kinds:
        children: Dict[int, Set[int]] = {}
        for rid, rec in region_table.items():
            if rec["kind"] == kind:
                children[rid] = set()
        children_by_kind[kind] = children

    for node, stack in node_regions.items():
        for i in range(len(stack) - 1):
            outer = stack[i]
            inner = stack[i + 1]

            outer_rec = region_table.get(outer)
            inner_rec = region_table.get(inner)
            if outer_rec is None or inner_rec is None:
                continue

            if outer_rec["kind"] != inner_rec["kind"]:
                continue

            kind = outer_rec["kind"]
            children_by_kind[kind][outer].add(inner)

    return children_by_kind


# ---------------------------------------------------------------------------
# regions_by_kind
# ---------------------------------------------------------------------------

def derive_regions_by_kind(
    region_table: Dict[int, Dict[str, Any]],
) -> Dict[str, List[int]]:
    """Group region ids by kind, each list sorted in ascending order.

    Args:
      region_table: rid -> region record dict (must contain "kind" key).

    Returns:
      {kind: sorted list of rids} for each distinct kind in region_table.
    """
    regions_by_kind: Dict[str, List[int]] = {}
    for rid, rec in region_table.items():
        kind = rec["kind"]
        regions_by_kind.setdefault(kind, []).append(rid)
    for rids in regions_by_kind.values():
        rids.sort()
    return regions_by_kind


# ---------------------------------------------------------------------------
# region_neighbors (RegionIO-based)
# ---------------------------------------------------------------------------

def _regionio_key(rio: RegionIO) -> Tuple[Any, int, Any, int]:
    """Return a canonical hashable key for a RegionIO edge.

    The key is (outside.node, outside.idx, inside.node, inside.idx).
    """
    return (rio.outside.node, rio.outside.idx, rio.inside.node, rio.inside.idx)


def derive_region_neighbors_from_region_views(
    region_views: Dict[int, RegionView],
) -> Dict[int, Set[int]]:
    """Derive region-level adjacency (undirected) from RegionViews.

    Two regions are neighbors if one's RegionView contains a RegionIO with
    key (A, B, C, D) and the other's contains the *reversed* key (C, D, A, B),
    i.e. they share the same edge but from opposite perspectives.

    Args:
      region_views: {rid: RegionView} mapping for all regions.

    Returns:
      {rid: set of neighboring rids} (undirected: if A is in neighbors[B]
      then B is in neighbors[A]).
    """
    # Map RegionIO key -> owning region id.
    rio_owner: Dict[Any, int] = {}
    for rid, rv in region_views.items():
        for rio in rv.inputs:
            rio_owner[_regionio_key(rio)] = rid
        for rio in rv.outputs:
            rio_owner[_regionio_key(rio)] = rid

    neighbors: Dict[int, Set[int]] = {rid: set() for rid in region_views}

    for rid, rv in region_views.items():
        for rio in rv.inputs:
            k = _regionio_key(rio)
            rk = (k[2], k[3], k[0], k[1])  # reversed key
            other = rio_owner.get(rk)
            if other is not None and other != rid:
                neighbors[rid].add(other)
                neighbors[other].add(rid)
        for rio in rv.outputs:
            k = _regionio_key(rio)
            rk = (k[2], k[3], k[0], k[1])  # reversed key
            other = rio_owner.get(rk)
            if other is not None and other != rid:
                neighbors[rid].add(other)
                neighbors[other].add(rid)

    return neighbors


# ---------------------------------------------------------------------------
# Optional helpers for passes
# ---------------------------------------------------------------------------

def derive_region_nodes_from_region_view(
    gm: Any,
    rv: RegionView,
) -> Set[Any]:
    """Derive the set of FX nodes belonging to a region from its RegionView.

    Does not require capture-time ``node_regions``; uses only the RegionView
    and the FX graph's user edges.

    Algorithm:
      1. Start from all inside endpoints of inputs (rv.inputs[*].inside.node).
      2. DFS forward along ``users`` in the FX graph.
      3. Stop traversal at output-inside endpoints (rv.outputs[*].inside.node).
      4. Always include all inside endpoints (inputs and outputs).

    Args:
      gm: FX GraphModule (used implicitly via node.users edges).
      rv: RegionView whose inside endpoints seed the traversal.

    Returns:
      Set of FX nodes that belong to this region.
    """
    input_inside  = {rio.inside.node for rio in rv.inputs}
    output_inside = {rio.inside.node for rio in rv.outputs}

    region_nodes: Set[Any] = set()
    stack = list(input_inside)

    while stack:
        node = stack.pop()
        if node in region_nodes:
            continue
        region_nodes.add(node)
        if node in output_inside:
            continue
        for consumer in list(node.users.keys()):
            if consumer not in region_nodes:
                stack.append(consumer)

    region_nodes |= input_inside
    region_nodes |= output_inside
    return region_nodes


def derive_module_names_for_quantize_region(
    gm: Any,
    region_table: Dict[int, Dict[str, Any]],
    children_by_kind: Dict[str, Dict[int, Set[int]]],
    region_views: Dict[int, RegionView],
    rid: int,
) -> List[str]:
    """Return sorted list of call_module targets in a quantize region.

    Excludes modules that belong exclusively to direct child quantize regions,
    so only the "own" (non-delegated) modules of ``rid`` are returned.

    Uses RegionView-based node derivation via ``derive_region_nodes_from_region_view``.
    Because ``build_region_views_post_export`` now iterates both the main graph
    and any HOp child sub-graphs (via ``_iter_all_nodes``), this correctly handles
    ``tta.quantize`` nested inside ``tta.autocast``.

    Args:
      gm:               FX GraphModule (post-export, with child sub-graphs).
      region_table:     ep._tta["region_table"].
      children_by_kind: ep._tta["children_by_kind"].
      region_views:     ep._tta["region_views"].
      rid:              Quantize region id of interest.

    Returns:
      Sorted list of module name strings owned by this region.

    Raises:
      KeyError:       If ``rid`` is not in ``region_table`` or ``region_views``.
      AssertionError: If ``rid`` is not a quantize region.
    """
    rec = region_table[rid]
    assert rec["kind"] == "quantize", (
        f"rid={rid} has kind={rec['kind']!r}; expected 'quantize'"
    )

    quant_children = children_by_kind.get("quantize", {})
    direct_children = quant_children.get(rid, set())

    region_nodes_cache: Dict[int, Set[Any]] = {}

    def _get_nodes(one_rid: int) -> Set[Any]:
        if one_rid not in region_nodes_cache:
            region_nodes_cache[one_rid] = derive_region_nodes_from_region_view(
                gm, region_views[one_rid]
            )
        return region_nodes_cache[one_rid]

    region_nodes = set(_get_nodes(rid))
    for child_rid in direct_children:
        region_nodes -= _get_nodes(child_rid)

    module_names = set()
    for node in region_nodes:
        if node.op == "call_module":
            # Pre-export FX graphs use call_module directly.
            module_names.add(str(node.target))
        elif node.op == "call_function":
            # torch.export inlines all module calls as aten ops; recover the
            # originating module from nn_module_stack metadata.
            # Each entry is (qualified_name, class_name_string).
            # Take the deepest non-root entry (most specific leaf module).
            stack = node.meta.get("nn_module_stack")
            if stack:
                for _key in reversed(list(stack.keys())):
                    qname, _cls = stack[_key]
                    if qname:  # skip root module (qname == '')
                        module_names.add(qname)
                        break

    return sorted(module_names)
