"""TTA-IR data types: Endpoint, RegionIO, RegionView, AnnotationIR.

IR Type Hierarchy
-----------------
The TTA intermediate representation is built in four layers, each adding a
higher-level abstraction over the FX graph:

  Endpoint
    The most primitive unit: a reference to one side of a single data-flow
    edge.  An Endpoint is a (node, idx) pair where ``node`` is an FX node and
    ``idx`` is the flattened argument position on that node (0 for producers).

  RegionIO
    One directed boundary *edge* crossing a region boundary, consisting of an
    ``outside`` Endpoint and an ``inside`` Endpoint.  For an input edge the
    outside endpoint is the producer (outside the region) and the inside
    endpoint is the consumer (inside the region); for an output edge the roles
    are reversed.  A RegionIO is always a single tensor crossing; a region with
    N inputs has N RegionIO objects in its ``inputs`` list.

  RegionView
    The complete boundary representation for *one* region: the full lists of
    input edges and output edges (each a RegionIO).  A RegionView is built
    post-export from the final FX graph and is the primary interface used by
    lowering and quantization passes.

  AnnotationIR
    The top-level IR object that ties everything together: the ExportedProgram,
    the FX GraphModule, the capture-time region_table, the per-node authority
    map, and all post-export derived structures (region_views, children_by_kind,
    etc.).  AnnotationIR is the single object passed between TTA passes.

Naming note
-----------
``region_discovery.RegionIO`` is a *different* class that holds the full IO
*set* for a lower_as region (a list of BoundaryTensor inputs and a list of
BoundaryTensor outputs).  It is imported in ``__init__.py`` as
``BoundaryRegionIO`` to avoid confusion.  See region_discovery.py for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch  # noqa: F401 — needed for forward-reference resolution of torch.fx.Node / torch.export.ExportedProgram annotations


@dataclass(frozen=True)
class Endpoint:
    """One end of a RegionIO boundary edge.

    Attributes:
      node: FX node (torch.fx.Node).
      idx:  Flattened argument position on the node.
            For producers (outside→inside edge): always 0.
            For consumers (inside→outside edge): flattened input index on that
            node as determined by depth-first traversal of node.args
            (tuples/lists flattened in order).
    """

    node: torch.fx.Node
    idx: int


@dataclass(frozen=True)
class RegionIO:
    """A single boundary edge crossing a region boundary.

    Each RegionIO represents *one tensor* flowing across the boundary of a
    region.  A region with N inputs has N RegionIO objects in its ``inputs``
    list; a region with M outputs has M RegionIO objects in its ``outputs``
    list.

    For an *input* edge (tensor entering the region):
      outside = producer endpoint (outside region, idx=0)
      inside  = consumer endpoint (inside region, idx=input_slot)

    For an *output* edge (tensor leaving the region):
      inside  = producer endpoint (inside region, idx=0)
      outside = consumer endpoint (outside region, idx=input_slot)

    Note: this class is distinct from ``region_discovery.RegionIO``
    (aliased as ``BoundaryRegionIO`` in the package ``__init__.py``), which
    holds *all* inputs and outputs for a lower_as region as lists of
    ``BoundaryTensor``.
    """

    outside: Endpoint
    inside: Endpoint


@dataclass
class RegionView:
    """Boundary representation for one region.

    Built post-export from the final FX graph.  Serves as the primary
    interface for lowering and quantization passes that need to inspect or
    modify the data flow at a region's boundary.

    Attributes:
      region_id: Unique integer identifier for the region.
      inputs:    Edges entering this region (outside→inside), one RegionIO
                 per tensor crossing the boundary.
      outputs:   Edges leaving this region (inside→outside), one RegionIO
                 per tensor crossing the boundary.
    """

    region_id: int
    inputs: List[RegionIO] = field(default_factory=list)
    outputs: List[RegionIO] = field(default_factory=list)


@dataclass
class AnnotationIR:
    """TTA annotation intermediate representation.

    The top-level IR object produced by the capture + post-export pipeline.
    It combines the ExportedProgram with all TTA metadata derived from the
    capture-time region stacks and post-export graph analysis, and is the
    single object threaded through TTA passes.

    Required fields (populated at construction):
      exported_program: torch.export.ExportedProgram — the exported model.
      gm:               FX GraphModule (ep.module()).
      region_table:     rid -> region record dict from capture state.
                        Each record has at minimum {"kind": str}.
      node_authority:   node -> {"quantize": rid|None, "autocast": rid|None, ...}
                        Maps each FX node to the innermost region of each kind
                        that contains it (None if no region of that kind).
      region_views:     rid -> RegionView; post-export boundary edges per region.

    Optional / derived fields (populated lazily or by specific passes):
      children_by_kind: kind -> {rid: set of direct child rids of same kind}.
      regions_by_kind:  kind -> sorted list of rids with that kind.
      region_neighbors: rid -> set of neighbor rids (undirected adjacency).
      quantize_evidence: rid -> effect dict (populated after ModelOpt).
      autocast_evidence: rid -> RegionAutocastEvidence (populated by enforcement).
      source_model:     Original nn.Module before export; needed by ModelOpt
                        quantization which cannot operate on ep.module().
    """

    exported_program: torch.export.ExportedProgram
    gm: torch.fx.GraphModule
    region_table: Dict[int, Dict[str, Any]]
    node_authority: Dict[torch.fx.Node, Dict[str, Optional[int]]]
    region_views: Dict[int, RegionView]
    children_by_kind: Dict[str, Dict[int, set]] = field(default_factory=dict)
    regions_by_kind: Dict[str, List[int]] = field(default_factory=dict)
    region_neighbors: Dict[int, set] = field(default_factory=dict)
    quantize_evidence: Optional[Dict[int, Dict]] = None
    autocast_evidence: Optional[Dict[int, Any]] = None
    # Original nn.Module before export; used by ModelOpt quantization which
    # cannot operate on ep.module() (a hollow GraphModule).
    source_model: Any = None
