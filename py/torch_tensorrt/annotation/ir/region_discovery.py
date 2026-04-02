"""Region discovery and boundary IO for lower_as regions.

This module operates on the *pre-export* FX graph (or post-export FX graph
with tta_regions metadata) to discover which nodes belong to each region and
to compute the boundary IO for a lower_as region.

Naming note
-----------
``BoundaryTensor`` and ``RegionIO`` here are *distinct* from the types in
``ir/types.py``:

  BoundaryTensor
    A (node, tensor_meta) pair representing one tensor on the boundary of a
    lower_as region.  Unlike ``ir.types.Endpoint``, which records an
    (node, idx) pair for edge-tracking in post-export RegionViews,
    BoundaryTensor carries the tensor metadata needed for plugin/impl
    signature checking.

  RegionIO  (this module; aliased as ``BoundaryRegionIO`` in the package)
    The *complete* IO set for a single lower_as region: a list of
    BoundaryTensor inputs and a list of BoundaryTensor outputs.  This is
    a region-level aggregate, whereas ``ir.types.RegionIO`` represents a
    single directed boundary edge (one tensor crossing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Set

from torch.fx import GraphModule, Node


@dataclass
class BoundaryTensor:
    """A tensor on the boundary of a lower_as region.

    Attributes:
      node:        The FX node that produces (for outputs) or supplies (for
                   inputs) this tensor.
      tensor_meta: Shape/dtype metadata from node.meta["tensor_meta"] or
                   node.meta["val"]; may be None if unavailable.

    Note: distinct from ``ir.types.Endpoint``.  BoundaryTensor captures
    tensor metadata for plugin/impl compatibility checks; Endpoint captures
    the flattened argument index for post-export RegionView edge tracking.
    """

    node: Node
    tensor_meta: Any


@dataclass
class RegionIO:
    """The complete boundary IO set for a single lower_as region.

    Aggregates all tensors that cross the region boundary, in the order they
    are first encountered during a scan of the region's nodes.

    Attributes:
      inputs:  Tensors produced *outside* the region and consumed inside it.
               One BoundaryTensor per unique outside-producer, in first-use
               order.
      outputs: Tensors produced *inside* the region and consumed outside it.
               One BoundaryTensor per unique escaping producer, in first-
               escape order.

    Note: aliased as ``BoundaryRegionIO`` in the package ``__init__.py`` to
    avoid confusion with ``ir.types.RegionIO``, which represents a single
    directed boundary edge (one tensor crossing with Endpoint fields).
    """

    inputs: List[BoundaryTensor]
    outputs: List[BoundaryTensor]


def collect_region_nodes(fx_module: GraphModule, region_id: int) -> List[Node]:
    """Collect all nodes that belong to the given region_id, sorted by tta_seq.

    Scans ``fx_module.graph.nodes`` for nodes whose ``tta_regions`` metadata
    list contains ``region_id``.

    Args:
      fx_module: The FX GraphModule to scan.
      region_id: The region identifier to collect nodes for.

    Returns:
      List of matching FX nodes sorted by their ``tta_seq`` metadata value
      (defaults to 0 when absent).
    """
    region_nodes: List[Node] = []
    for n in fx_module.graph.nodes:
        regions = n.meta.get("tta_regions", [])
        if regions and region_id in regions:
            region_nodes.append(n)
    region_nodes.sort(key=lambda n: n.meta.get("tta_seq", 0))
    return region_nodes


def compute_region_io(
    fx_module: GraphModule,
    region_nodes: Sequence[Node],
) -> RegionIO:
    """Compute the boundary IO for a set of region nodes.

    Inputs are outside-producers in first-use order; outputs are escaping
    inside-producers in first-escape order.

    Args:
      fx_module:    The FX GraphModule (used only for graph-order context).
      region_nodes: Ordered sequence of FX nodes belonging to the region.

    Returns:
      RegionIO with inputs = outside producers (first-use order) and
      outputs = escaping inside producers (first-escape order).
    """
    region_set: Set[Node] = set(region_nodes)
    inputs: List[BoundaryTensor] = []
    seen_input_nodes: Set[Node] = set()
    outputs: List[BoundaryTensor] = []
    seen_output_nodes: Set[Node] = set()

    for n in region_nodes:
        for arg in n.all_input_nodes:
            if arg in region_set or arg in seen_input_nodes:
                continue
            seen_input_nodes.add(arg)
            inputs.append(
                BoundaryTensor(
                    node=arg,
                    tensor_meta=arg.meta.get("tensor_meta") or arg.meta.get("val"),
                )
            )

    for n in region_nodes:
        if n in seen_output_nodes:
            continue
        if any(u not in region_set for u in n.users):
            seen_output_nodes.add(n)
            outputs.append(
                BoundaryTensor(
                    node=n,
                    tensor_meta=n.meta.get("tensor_meta") or n.meta.get("val"),
                )
            )

    return RegionIO(inputs=inputs, outputs=outputs)
