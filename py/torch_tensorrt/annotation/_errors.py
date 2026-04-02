"""TTA Structured Error Types.

This module defines the exception hierarchy for the TTA (Torch-TensorRT
Annotation) layer.  All TTA failure paths raise one of these types so that
callers can inspect structured context fields without parsing message strings.

Hierarchy
---------
RuntimeError
└── TTADiagnosticError          # base: stage + leaf_op + impl_id

Design note
-----------
All errors are *stage-labeled*: the ``stage`` attribute is one of ``"export"``,
``"lowering"``, or ``"build"``, matching the TTA pipeline stage in which the
failure occurred.  ``leaf_op`` identifies the annotated op and ``impl_id``
identifies the implementation.
"""

from __future__ import annotations

from typing import Optional


class TTADiagnosticError(RuntimeError):
    """Structured diagnostic error for the TTA annotation layer.

    All TTA failure paths raise this class or one of its subclasses so that
    callers can programmatically inspect ``stage``, ``leaf_op``, and
    ``impl_id`` without parsing the human-readable message string.

    Parameters
    ----------
    msg:
        Human-readable description of the failure.
    stage:
        Pipeline stage in which the error occurred.  One of ``"export"``,
        ``"lowering"``, or ``"build"``.
    leaf_op:
        Qualified op name that triggered the error.  May be ``None``.
    impl_id:
        Implementation identity string.  May be ``None``.

    Attributes
    ----------
    stage : str
    leaf_op : str or None
    impl_id : str or None
    message : str
    """

    def __init__(
        self,
        msg: str,
        *,
        stage: str,
        leaf_op: Optional[str] = None,
        impl_id: Optional[str] = None,
    ) -> None:
        self.stage = stage
        self.leaf_op = leaf_op
        self.impl_id = impl_id
        self.message = msg
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts: list[str] = [f"[TTA {self.stage.upper()}]"]
        if self.leaf_op:
            parts.append(self.leaf_op)
        if self.impl_id:
            parts.append(f"(impl={self.impl_id})")
        return f"{' '.join(parts)}: {self.message}"

    def __str__(self) -> str:
        return self._format_message()
