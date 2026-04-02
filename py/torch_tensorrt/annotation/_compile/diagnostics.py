"""Diagnostic types and error-raising helpers for require=True enforcement."""

from dataclasses import dataclass, field
from typing import Any, Dict, NoReturn, Optional


@dataclass
class RegionReport:
    """
    Per-region summary of realized effects for tta.quantize and tta.autocast.

    region_id: Integer region id.
    kind:      Region kind string ("quantize" or "autocast").
    name:      Optional user-supplied name from tta.quantize/autocast(..., name=...).
    effect:    Summary string:
               - "applied"     -> transform had a material effect.
               - "zero_effect" -> eligible ops/modules found but no material effect.
               - "skipped"     -> transform skipped (e.g. overlap constraint).
               - "unknown"     -> evidence not computed or kind not handled.
    details:   Optional extra context dict.
               For quantize: may include {"quantized_modules": [...], "mode": "int8"}.
               For autocast: may include {"cast_ops_inserted": N, "mode": "fp16"}.
    """
    region_id: int
    kind: str
    name: Optional[str]
    effect: str
    details: Dict[str, Any] = field(default_factory=dict)


class RegionFailure(RuntimeError):
    """
    Structured description of a require=True region failure.

    stage:            Compilation stage name (e.g. "AutocastGraphCheck", "ModelOptQuantize").
    region_id:        Integer region id.
    kind:             Region kind string ("quantize" or "autocast").
    name:             Optional user-supplied name from tta.quantize/autocast(..., name=...).
    reason:           Deterministic reason code string:
                      - "NO_ELIGIBLE_OPS"            (autocast: no float ops in region)
                      - "ZERO_EFFECT"                (autocast/quantize: eligible ops but no casts/quant)
                      - "NO_ELIGIBLE_MODULES"        (quantize: no call_module nodes in region)
                      - "SKIPPED_OVERLAP_CONSTRAINT" (quantize: work skipped due to overlap)
                      - "MISSING_EVIDENCE"           (internal: evidence not computed)
    actionable_reason: Human-readable explanation of why the check failed.
    remediation:      Suggested fix for the failure.
    """

    def __init__(
        self,
        stage: str,
        region_id: int,
        kind: str,
        name: Optional[str],
        reason: str,
        *,
        actionable_reason: Optional[str] = None,
        remediation: Optional[str] = None,
    ) -> None:
        self.stage = stage
        self.region_id = region_id
        self.kind = kind
        self.name = name
        self.reason = reason
        self.actionable_reason = actionable_reason
        self.remediation = remediation

        name_part = f"Name={name!r}" if name is not None else "Name=<unnamed>"
        lines = [
            f"TTA require=True not satisfied: "
            f"Stage={stage} Region={kind}:{region_id} "
            f"{name_part} Reason={reason}",
        ]
        if actionable_reason:
            lines.append(f"  Actionable: {actionable_reason}")
        if remediation:
            lines.append(f"  Remediation: {remediation}")
        super().__init__("\n".join(lines))


def raise_region_failure(f: RegionFailure) -> NoReturn:
    """Raise a RegionFailure (already a RuntimeError subclass)."""
    raise f
