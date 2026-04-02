"""RegionView repair utilities for post-transform boundary updates.

After a transform changes the internal structure of a region, the RegionView
endpoints that point inside the region may become stale.  Use the helpers
here to repair them without rerunning the full post-export build.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Set

from .types import Endpoint, RegionIO, RegionView


IsInRegion = Callable[[Any], bool]
IsInRegionById = Callable[[int, Any], bool]


def _is_fx_node(v: Any) -> bool:
    """Return True if ``v`` looks like an FX node (duck-typed)."""
    return hasattr(v, "op") and hasattr(v, "args") and hasattr(v, "users")


def iter_flat_inputs(consumer: Any) -> List[Any]:
    """Flatten consumer.args to an ordered list of FX nodes.

    Traverses ``consumer.args`` depth-first, recursing into tuples and lists.
    Must stay consistent with ``region_view._flatten_input_occurrences``.

    Args:
      consumer: An FX node whose args may contain nested tuples/lists.

    Returns:
      Flat list of FX nodes in depth-first, left-to-right order.
    """
    flat: List[Any] = []

    def collect(val: Any) -> None:
        if _is_fx_node(val):
            flat.append(val)
        elif isinstance(val, (tuple, list)):
            for x in val:
                collect(x)

    collect(consumer.args)
    return flat


def find_input_index(consumer: Any, producer: Any) -> int:
    """Return the flattened input index of ``producer`` in ``consumer.args``.

    Args:
      consumer: The FX node to search within.
      producer: The FX node to find.

    Returns:
      The flattened index (consistent with ``iter_flat_inputs``) at which
      ``producer`` first appears in ``consumer.args``.

    Raises:
      RuntimeError: If ``producer`` is not found in ``consumer.args``.
    """
    for idx, n in enumerate(iter_flat_inputs(consumer)):
        if n is producer:
            return idx
    raise RuntimeError(
        f"producer {producer!r} not found in args of consumer {consumer!r}; "
        f"cannot determine flattened input index"
    )


def gather_neighbor_regions(
    target_rid: int,
    region_views: Dict[int, RegionView],
) -> Set[int]:
    """Identify regions that share a boundary edge with ``target_rid``.

    Two regions are neighbors when one's output consumer/idx matches the
    other's input consumer/idx (or vice-versa).

    Args:
      target_rid:   The region whose neighbors to find.
      region_views: Full {rid: RegionView} mapping.

    Returns:
      Set of region ids that share at least one boundary edge with
      ``target_rid``.  Does not include ``target_rid`` itself.
    """
    neighbors: Set[int] = set()
    target = region_views[target_rid]

    # Downstream neighbors: target.outputs → other's inputs share consumer+idx
    for out_rio in target.outputs:
        out_consumer = out_rio.outside.node
        out_idx = out_rio.outside.idx
        for rid, rv in region_views.items():
            if rid == target_rid:
                continue
            for in_rio in rv.inputs:
                if in_rio.inside.node is out_consumer and in_rio.inside.idx == out_idx:
                    neighbors.add(rid)

    # Upstream neighbors: target.inputs ← other's outputs share producer+idx
    for in_rio in target.inputs:
        in_producer = in_rio.outside.node
        for rid, rv in region_views.items():
            if rid == target_rid:
                continue
            for out_rio in rv.outputs:
                if out_rio.inside.node is in_producer:
                    neighbors.add(rid)

    return neighbors


def recompute_inside_from_outside(
    rv: RegionView,
    is_in_region: IsInRegion,
) -> None:
    """Repair RegionView for a transformed region.

    Keeps "outside" endpoints fixed; re-discovers "inside" endpoints from
    the updated FX graph using ``is_in_region(node) -> bool``.

    Mutates ``rv`` in place.

    Args:
      rv:           The RegionView to repair (mutated in place).
      is_in_region: Predicate returning True if a node belongs to this region.

    Raises:
      RuntimeError: If an inside endpoint cannot be found in the updated graph
                    (e.g. the transform broke a required data-flow connection).
    """
    # Inputs: outside producer → inside consumer
    for i, rio in enumerate(rv.inputs):
        outside_prod = rio.outside.node
        new_inside_consumer = None
        new_inside_idx = None

        for consumer in list(outside_prod.users.keys()):
            if is_in_region(consumer):
                new_inside_consumer = consumer
                new_inside_idx = find_input_index(consumer, outside_prod)
                break

        if new_inside_consumer is None:
            raise RuntimeError(
                f"recompute_inside_from_outside: no inside consumer found for "
                f"outside producer {outside_prod!r} (region_id={rv.region_id}, "
                f"input slot {i}); the transform may have broken this data-flow edge"
            )

        rv.inputs[i] = RegionIO(
            outside=rio.outside,
            inside=Endpoint(node=new_inside_consumer, idx=new_inside_idx),
        )

    # Outputs: inside producer → outside consumer
    for i, rio in enumerate(rv.outputs):
        outside_consumer = rio.outside.node
        new_inside_prod = None

        for prod in outside_consumer.all_input_nodes:
            if is_in_region(prod):
                new_inside_prod = prod
                break

        if new_inside_prod is None:
            raise RuntimeError(
                f"recompute_inside_from_outside: no inside producer found feeding "
                f"outside consumer {outside_consumer!r} (region_id={rv.region_id}, "
                f"output slot {i}); the transform may have broken this data-flow edge"
            )

        rv.outputs[i] = RegionIO(
            inside=Endpoint(node=new_inside_prod, idx=0),
            outside=rio.outside,
        )


def recompute_outside_from_inside(
    rv: RegionView,
    is_in_region: IsInRegion,
) -> None:
    """Repair RegionView for a neighbor region.

    Keeps "inside" endpoints fixed; re-discovers "outside" endpoints from
    the updated FX graph using ``is_in_region(node) -> bool``.

    Mutates ``rv`` in place.

    Args:
      rv:           The RegionView to repair (mutated in place).
      is_in_region: Predicate returning True if a node belongs to this neighbor
                    region (not the transformed region).

    Raises:
      RuntimeError: If an outside endpoint cannot be found in the updated graph.
    """
    # Inputs: outside producer → inside consumer
    for i, rio in enumerate(rv.inputs):
        inside_consumer = rio.inside.node
        new_outside_prod = None
        new_outside_idx = None

        for prod in inside_consumer.all_input_nodes:
            if not is_in_region(prod):
                new_outside_prod = prod
                new_outside_idx = find_input_index(inside_consumer, prod)
                break

        if new_outside_prod is None:
            raise RuntimeError(
                f"recompute_outside_from_inside: no outside producer found for "
                f"inside consumer {inside_consumer!r} (region_id={rv.region_id}, "
                f"input slot {i}); the transform may have broken this data-flow edge"
            )

        rv.inputs[i] = RegionIO(
            outside=Endpoint(node=new_outside_prod, idx=new_outside_idx),
            inside=rio.inside,
        )

    # Outputs: inside producer → outside consumer
    for i, rio in enumerate(rv.outputs):
        inside_prod = rio.inside.node
        new_outside_consumer = None
        new_outside_idx = None

        for consumer in list(inside_prod.users.keys()):
            if not is_in_region(consumer):
                new_outside_consumer = consumer
                new_outside_idx = find_input_index(consumer, inside_prod)
                break

        if new_outside_consumer is None:
            raise RuntimeError(
                f"recompute_outside_from_inside: no outside consumer found for "
                f"inside producer {inside_prod!r} (region_id={rv.region_id}, "
                f"output slot {i}); the transform may have broken this data-flow edge"
            )

        rv.outputs[i] = RegionIO(
            inside=rio.inside,
            outside=Endpoint(node=new_outside_consumer, idx=new_outside_idx),
        )


def repair_region_views_after_transform(
    *,
    region_views: Dict[int, RegionView],
    target_rid: int,
    is_in_target: IsInRegion,
    is_in_neighbor: IsInRegionById,
) -> None:
    """Repair RegionView[target_rid] and all neighboring regions after a graph transform.

    After a transform that changes the internals of ``target_rid``, this
    function repairs the stale inside endpoints in the transformed region's
    RegionView (using the fixed outside endpoints as anchors) and then repairs
    the outside endpoints of all neighboring regions (using their fixed inside
    endpoints as anchors).

    Mutates ``region_views`` in place.

    Args:
      region_views:   ep._tta["region_views"] (mutated in place).
      target_rid:     The region that was transformed.
      is_in_target:   ``node -> bool``; True if the node now belongs to
                      ``target_rid`` in the updated graph.
      is_in_neighbor: ``(rid, node) -> bool``; True if ``node`` belongs to
                      region ``rid`` in the updated graph.

    Raises:
      RuntimeError: Propagated from ``recompute_inside_from_outside`` or
                    ``recompute_outside_from_inside`` if any endpoint cannot
                    be re-discovered.
    """
    neighbors = gather_neighbor_regions(target_rid, region_views)

    recompute_inside_from_outside(region_views[target_rid], is_in_target)

    for rid in neighbors:
        recompute_outside_from_inside(
            region_views[rid],
            lambda node, r=rid: is_in_neighbor(r, node),
        )

