"""TTA intermediate representation types and construction helpers.

Public API
----------
Types (from ``ir.types``):
  Endpoint         — one end of a boundary edge: (node, idx)
  RegionIO         — a single directed boundary edge: (outside, inside) Endpoint pair
  RegionView       — full boundary for one region: lists of input/output RegionIO edges
  AnnotationIR     — top-level IR object tying together all TTA metadata

Region discovery (from ``ir.region_discovery``; for lower_as boundary IO):
  BoundaryTensor   — (node, tensor_meta) pair on a lower_as region boundary
  BoundaryRegionIO — full IO set for a lower_as region (list of BoundaryTensor inputs/outputs)
  collect_region_nodes — collect FX nodes belonging to a region by tta_regions metadata
  compute_region_io    — compute BoundaryRegionIO for a sequence of region nodes

Post-export construction (from ``ir.region_view``):
  build_region_views_post_export          — build RegionViews from the exported FX graph
  derive_region_children_by_kind          — same-kind parent→child relationships
  derive_regions_by_kind                  — group region ids by kind
  derive_region_neighbors_from_region_views — undirected region adjacency
  derive_region_nodes_from_region_view    — node set for a region from its RegionView
  derive_module_names_for_quantize_region — module names owned by a quantize region

Authority (from ``ir.authority``):
  derive_node_authority — per-node innermost-region authority per kind

Repair (from ``ir.boundary_repair``):
  repair_region_views_after_transform — repair RegionViews after a graph transform
  recompute_inside_from_outside       — repair transformed-region inside endpoints
  recompute_outside_from_inside       — repair neighbor-region outside endpoints
  gather_neighbor_regions             — find regions sharing a boundary edge
  iter_flat_inputs                    — flatten node.args to ordered list of FX nodes
  find_input_index                    — flattened arg index of a producer in a consumer

Validation (from ``ir.boundary_validation``):
  io_compatible — check region boundary IO against an impl descriptor
"""

from .types import AnnotationIR, Endpoint, RegionIO, RegionView
from .authority import derive_node_authority
from .region_view import (
    build_region_views_post_export,
    derive_region_children_by_kind,
    derive_regions_by_kind,
    derive_region_neighbors_from_region_views,
    derive_region_nodes_from_region_view,
    derive_module_names_for_quantize_region,
)
from .region_discovery import (
    BoundaryTensor,
    RegionIO as BoundaryRegionIO,
    collect_region_nodes,
    compute_region_io,
)
from .boundary_validation import io_compatible
from .boundary_repair import (
    repair_region_views_after_transform,
    recompute_inside_from_outside,
    recompute_outside_from_inside,
    gather_neighbor_regions,
    iter_flat_inputs,
    find_input_index,
)

__all__ = [
    # Core IR types
    "Endpoint",
    "RegionIO",
    "RegionView",
    "AnnotationIR",
    # Authority
    "derive_node_authority",
    # Post-export construction
    "build_region_views_post_export",
    "derive_region_children_by_kind",
    "derive_regions_by_kind",
    "derive_region_neighbors_from_region_views",
    "derive_region_nodes_from_region_view",
    "derive_module_names_for_quantize_region",
    # Region discovery (lower_as boundary IO)
    "BoundaryTensor",
    "BoundaryRegionIO",
    "collect_region_nodes",
    "compute_region_io",
    # Validation
    "io_compatible",
    # Repair
    "repair_region_views_after_transform",
    "recompute_inside_from_outside",
    "recompute_outside_from_inside",
    "gather_neighbor_regions",
    "iter_flat_inputs",
    "find_input_index",
]
