"""TTA compilation pipeline: TTA-IR construction.

This module sits at the core of ``tta.compile()``.  After ``torch.export.export``
has traced the model inside a TTA capture context, this module runs the following
pipeline stage:

Stage 1 — TTA-IR construction (``build_annotation_ir``)
    Collects the transient ``node.meta["tta_regions"]`` tags written during graph
    tagging, derives authority (innermost region per kind per node), builds
    ``RegionView`` boundary structures, and assembles the ``AnnotationIR``.  All
    transient node-meta keys are removed; the sole authoritative post-capture
    representation is the ``RegionView`` boundary stored in ``ep._tta``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..ir.authority import derive_node_authority
from ..ir.region_view import (
    build_region_views_post_export,
    derive_region_children_by_kind,
    derive_regions_by_kind,
    derive_region_neighbors_from_region_views,
)
from ..ir.types import AnnotationIR

logger = logging.getLogger(__name__)

NODE_META_REGIONS_KEY = "tta_regions"


# ---------------------------------------------------------------------------
# TTA-IR construction
# ---------------------------------------------------------------------------

def build_annotation_ir(
    exported_program: Any,
    gm: Any,
    region_table: Dict[int, Dict[str, Any]],
) -> AnnotationIR:
    """
    Build AnnotationIR from the exported FX graph and region_table.

    Steps:
      1. Collect node_regions from node.meta[NODE_META_REGIONS_KEY].
      2. Derive node_authority (innermost region per kind per node).
      3. Build RegionViews (boundary edges).
      4. Derive children_by_kind, regions_by_kind, region_neighbors.
      5. Store in AnnotationIR and (optionally) in ep._tta.

    Args:
      exported_program: torch.export.ExportedProgram.
      gm:               ep.module() FX GraphModule.
      region_table:     Snapshot from _capture_state.get_region_table().

    Returns:
      Populated AnnotationIR.
    """
    # 1. Collect transient node_regions map from node.meta, then immediately
    #    delete the transient keys from every node.  After this point, all
    #    tta region information lives exclusively in ep._tta (via RegionViews),
    #    not in node.meta (transient).  Node meta is never stored or recovered.
    node_regions: Dict[Any, List[int]] = {}

    def _collect_from_graph(graph: Any) -> None:
        for node in graph.nodes:
            stack = node.meta.get(NODE_META_REGIONS_KEY)
            if stack:
                node_regions[node] = list(stack)
            node.meta.pop(NODE_META_REGIONS_KEY, None)
            node.meta.pop("tta_seq", None)

    _collect_from_graph(gm.graph)
    for _child_name, child_mod in gm.named_children():
        if hasattr(child_mod, "graph"):
            _collect_from_graph(child_mod.graph)

    # 2. Derive authority.
    node_authority = derive_node_authority(
        node_regions=node_regions,
        region_table=region_table,
    )

    # 3. Build RegionViews.
    region_ids = sorted(region_table.keys())
    region_views = build_region_views_post_export(
        gm,
        node_regions=node_regions,
        region_ids=region_ids,
    )

    # 4. Derived indexes.
    children_by_kind = derive_region_children_by_kind(node_regions, region_table)
    regions_by_kind = derive_regions_by_kind(region_table)
    region_neighbors = derive_region_neighbors_from_region_views(region_views)

    ann_ir = AnnotationIR(
        exported_program=exported_program,
        gm=gm,
        region_table=region_table,
        node_authority=node_authority,
        region_views=region_views,
        children_by_kind=children_by_kind,
        regions_by_kind=regions_by_kind,
        region_neighbors=region_neighbors,
    )

    # 5. Persist into ep._tta (alongside any existing lower_as data).
    _persist_to_ep(exported_program, ann_ir)

    return ann_ir


def _persist_to_ep(
    exported_program: Any,
    ann_ir: AnnotationIR,
) -> None:
    """Store TTA-IR structures in ep._tta, merging with any existing data.

    Only boundary-level structures are persisted.  Node-level transient data
    (tta_regions, tta_seq) is never stored — TTA is region-oriented; the
    RegionView boundary is the sole authoritative post-capture representation.
    """
    existing = getattr(exported_program, "_tta", {}) or {}
    existing["region_table"] = ann_ir.region_table
    existing["region_views"] = ann_ir.region_views
    existing["children_by_kind"] = ann_ir.children_by_kind
    existing["regions_by_kind"] = ann_ir.regions_by_kind
    existing["region_neighbors"] = ann_ir.region_neighbors
    exported_program._tta = existing
