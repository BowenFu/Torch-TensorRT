"""require=True enforcement for tta.quantize and tta.autocast regions.

Enforcement runs after ModelOpt and before lowering.  It checks evidence
gathered during ModelOpt transforms (quantize) or from the exported graph
(autocast AMP-based) and raises RegionFailure for any region that has
require=True but shows no material effect.

Current coverage:
    - tta.quantize: checks that at least one eligible module was quantized.
    - tta.autocast: checks that autocast-like cast ops appear in the region.
"""

from __future__ import annotations

from typing import Any, Dict

from .diagnostics import RegionFailure, raise_region_failure

import torch


# ---------------------------------------------------------------------------
# Autocast evidence
# ---------------------------------------------------------------------------

class RegionAutocastEvidence:
    """Per-region summary of autocast-like cast ops."""
    __slots__ = ("effect_casts", "no_candidate_ops")

    def __init__(self, effect_casts: int = 0, no_candidate_ops: bool = True) -> None:
        self.effect_casts = effect_casts
        # True means "no float ops encountered yet"; becomes False when first float op is seen.
        self.no_candidate_ops = no_candidate_ops


def _is_autocast_like_cast(node: Any) -> bool:
    """Return True if node is an ATen cast op inserted by torch.amp.autocast.

    Before run_decompositions(): checks for wrap_with_autocast higher-order fn.
    After  run_decompositions(): checks for aten._to_copy / aten.to.* cast ops.
    """
    if node.op != "call_function":
        return False
    tgt = node.target
    try:
        name = str(tgt)
    except (AttributeError, TypeError):
        return False
    # Post-decomposition: explicit cast ops.
    if name.startswith("aten._to_copy") or name.startswith("aten.to."):
        return True
    # Pre-decomposition: torch.amp.autocast wraps the region.
    if "wrap_with_autocast" in name or "autocast" in name.lower():
        return True
    return False


def _is_potential_autocast_eligible_op(node: Any) -> bool:
    """Return True if node operates on a floating-point tensor (autocast candidate).

    Also returns True for wrap_with_autocast nodes (pre-decomposition graph).
    """
    # Pre-decomposition: wrap_with_autocast node is inherently eligible.
    if node.op == "call_function":
        try:
            name = str(node.target)
        except (AttributeError, TypeError):
            name = ""
        if "wrap_with_autocast" in name:
            return True

    # Post-decomposition: inspect dtype from node.meta.
    val = node.meta.get("val", None)
    if val is None or not hasattr(val, "dtype"):
        return False
    dt = val.dtype
    return bool(getattr(dt, "is_floating_point", False))


def compute_autocast_evidence(ann_ir: Any) -> Dict[int, RegionAutocastEvidence]:
    """
    Infer per-region autocast "effect" from the exported FX graph.

    Counts:
      effect_casts:    number of autocast-like ATen cast ops attributed to the region.
      no_candidate_ops: True if the region has no floating-point ops at all.

    Args:
      ann_ir: AnnotationIR (from ir/types.py)
    """
    gm = ann_ir.gm
    region_table = ann_ir.region_table
    node_auth = ann_ir.node_authority

    ev: Dict[int, RegionAutocastEvidence] = {}
    for rid, rinfo in region_table.items():
        if rinfo["kind"] != "autocast":
            continue
        ev[rid] = RegionAutocastEvidence()

    for node in gm.graph.nodes:
        auth = node_auth.get(node, {})
        rid = auth.get("autocast")
        if rid is None or rid not in ev:
            continue

        if _is_autocast_like_cast(node):
            ev[rid].effect_casts += 1

        if _is_potential_autocast_eligible_op(node):
            ev[rid].no_candidate_ops = False

    # Second pass: handle wrap_with_autocast HOp nodes that are not directly
    # tagged in node_authority (torch.export creates them outside our region
    # stack push/pop), but whose getitem successors ARE tagged via sub-graph
    # tracing.  The presence of a wrap_with_autocast whose outputs feed a
    # tagged region is reliable evidence that autocast had effect.
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        try:
            name = str(node.target)
        except (AttributeError, TypeError):
            name = ""
        if "wrap_with_autocast" not in name:
            continue
        for user in node.users:
            auth = node_auth.get(user, {})
            rid = auth.get("autocast")
            if rid is not None and rid in ev:
                ev[rid].effect_casts += 1
                ev[rid].no_candidate_ops = False
                break  # count once per wrap_with_autocast node

    return ev


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

def enforce_require(ann_ir: Any) -> None:
    """
    Enforce require=True for all quantize and autocast regions.

    Raises RuntimeError (via raise_region_failure) for the first failing region.
    """
    _enforce_autocast_require(ann_ir)
    _enforce_quantize_require(ann_ir)


def _enforce_autocast_require(ann_ir: Any) -> None:
    """Check require=True for all autocast regions; raise RegionFailure on the first violation.

    Computes autocast evidence from the exported graph and verifies that each
    ``require=True`` autocast region has at least one cast op inserted (material
    effect).  Raises on the first region that fails, in region-id order.
    """
    evidence = compute_autocast_evidence(ann_ir)

    for rid, rinfo in ann_ir.region_table.items():
        if rinfo["kind"] != "autocast":
            continue
        cfg = rinfo.get("args", {})
        if not cfg.get("require", False):
            continue

        ev = evidence.get(rid)
        if ev is None:
            raise_region_failure(RegionFailure(
                "AutocastGraphCheck", rid, "autocast", rinfo.get("name"), "MISSING_EVIDENCE",
                actionable_reason="Internal error: autocast evidence was not computed for this region.",
                remediation="This is an internal TTA error; please file a bug report.",
            ))

        if ev.effect_casts == 0 and ev.no_candidate_ops:
            raise_region_failure(RegionFailure(
                "AutocastGraphCheck", rid, "autocast", rinfo.get("name"), "NO_ELIGIBLE_OPS",
                actionable_reason=(
                    "No operations eligible for autocast were found in the region."
                ),
                remediation=(
                    "Use mode='fp16' or 'bf16' for compute-heavy ops (linear, conv). "
                    "mode='disable' regions cannot have require=True."
                ),
            ))

        if ev.effect_casts == 0 and not ev.no_candidate_ops:
            raise_region_failure(RegionFailure(
                "AutocastGraphCheck", rid, "autocast", rinfo.get("name"), "ZERO_EFFECT",
                actionable_reason=(
                    "No cast operations were inserted — the region may contain no fp32 ops "
                    "eligible for fp16/bf16 casting."
                ),
                remediation=(
                    "Verify the model runs fp32 ops (matmul, conv) inside the region. "
                    "For bf16 support, ensure hardware and TRT version support it."
                ),
            ))


def _enforce_quantize_require(ann_ir: Any) -> None:
    """Check require=True for all quantize regions; raise RegionFailure on the first violation.

    Reads ``ann_ir.quantize_evidence`` (populated by ``run_modelopt_quantization``)
    and verifies that each ``require=True`` quantize region was neither skipped due
    to an overlap constraint nor resulted in zero quantization effect.  Raises on
    the first region that fails, in region-id order.
    """
    quant_evidence: Dict[int, Dict] = ann_ir.quantize_evidence or {}

    for rid, rinfo in ann_ir.region_table.items():
        if rinfo["kind"] != "quantize":
            continue
        cfg = rinfo.get("args", {})
        if not cfg.get("require", False):
            continue

        ev = quant_evidence.get(rid, {})

        if ev.get("skipped_overlap", False):
            raise_region_failure(RegionFailure(
                "ModelOptQuantize", rid, "quantize", rinfo.get("name"),
                "SKIPPED_OVERLAP_CONSTRAINT",
                actionable_reason=(
                    "Region was skipped due to overlap with another quantize region."
                ),
                remediation=(
                    "Ensure quantize regions do not overlap or nest. "
                    "Use separate regions for each module."
                ),
            ))

        if ev.get("no_eligible_modules", False):
            raise_region_failure(RegionFailure(
                "ModelOptQuantize", rid, "quantize", rinfo.get("name"),
                "NO_ELIGIBLE_MODULES",
                actionable_reason=(
                    "No eligible PyTorch modules (nn.Linear, nn.Conv2d, etc.) were found "
                    "inside the region."
                ),
                remediation=(
                    "Ensure the region boundary contains module calls, not raw functional ops "
                    "(use nn.Linear not F.linear)."
                ),
            ))

        if ev.get("effect", 0) == 0:
            raise_region_failure(RegionFailure(
                "ModelOptQuantize", rid, "quantize", rinfo.get("name"),
                "ZERO_EFFECT",
                actionable_reason=(
                    "No quantization was applied — the region may contain no eligible modules "
                    "(conv, linear, matmul)."
                ),
                remediation=(
                    "Move the region boundary to include nn.Linear or nn.Conv2d modules, "
                    "or use mode='auto' to let TRT pick."
                ),
            ))
