"""Derive per-node authoritative region for each annotation kind.

Authority rule: for each node and each kind (quantize, autocast, ...),
the authoritative region is the *innermost* region of that kind in the
node's membership stack (outermost → innermost order).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def derive_node_authority(
    *,
    node_regions: Dict[Any, List[int]],
    region_table: Dict[int, Dict[str, Any]],
) -> Dict[Any, Dict[str, Optional[int]]]:
    """Compute the authoritative region per kind for each node.

    For each node, scans its membership stack (outermost→innermost) and
    records the *last* (innermost) region of each kind seen.  A node that
    belongs to no region of a given kind gets ``None`` for that kind.

    Args:
      node_regions: Mapping from FX node to its region membership stack,
                    ordered outermost→innermost (e.g. [r0, r1, r2]).
      region_table: Mapping from region id to its record dict, which must
                    contain at least a ``"kind"`` key.

    Returns:
      Mapping ``{node: {kind: rid | None}}`` for all nodes in
      ``node_regions`` and all kinds present in ``region_table``.
      Additional kinds beyond quantize/autocast are captured automatically
      if they appear in ``region_table``.
    """
    # Collect all kinds that appear in the table.
    all_kinds = {rec["kind"] for rec in region_table.values()}

    out: Dict[Any, Dict[str, Optional[int]]] = {}
    for node, stack in node_regions.items():
        authority: Dict[str, Optional[int]] = {k: None for k in all_kinds}
        for rid in stack:
            rec = region_table.get(rid)
            if rec is None:
                continue
            kind = rec["kind"]
            authority[kind] = rid  # innermost wins (last assignment)
        out[node] = authority
    return out
