"""TTA compilation pipeline: TTA-IR construction.

Public API exported from this package:

Pipeline entry points:
    build_annotation_ir  — Construct AnnotationIR from an exported FX graph.
"""

from .pipeline import build_annotation_ir

__all__ = [
    "build_annotation_ir",
]
