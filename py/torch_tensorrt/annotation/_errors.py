"""TTA Structured Error Types.

This module defines the exception hierarchy for the TTA (Torch-TensorRT
Annotation) layer.  All TTA failure paths raise one of these types so that
callers can inspect structured context fields without parsing message strings.

Hierarchy
---------
RuntimeError
└── TTADiagnosticError          # base: stage + leaf_op + impl_id
    ├── TTABuiltinError          # lowering via network.add_* methods
    ├── TTAPluginError           # lowering via TRT plugin registry
    └── LowerAsError             # lower_as pass: region-level failures

When to use each type
---------------------
TTADiagnosticError
    Use directly only in the annotation entry-point (``__init__.py``) when the
    failure does not belong to a more specific category, e.g. op-registration
    errors during ``torch.export``.

TTABuiltinError
    Raised by ``_builtin_lowering.py`` when a :class:`BuiltinSpec` cannot be
    lowered to a TRT built-in layer (method not found, argument binding error,
    or TRT API rejection).

TTAPluginError
    Raised by ``_plugin_lowering.py`` when a :class:`RegistryPluginSpec` cannot be
    lowered via the TRT plugin registry (plugin not registered, field
    population failure, or layer-creation failure).

LowerAsError
    Raised by ``_lower_as_pass.py`` when the ``tta.lower_as`` FX pass cannot
    lower a region (unresolved impl_id, IO mismatch, or op-materialisation
    failure).  Carries additional region-level fields (``rid``, ``name``,
    ``reason``) beyond the base class.

Design note
-----------
All errors are *stage-labeled*: the ``stage`` attribute is one of ``"export"``,
``"lowering"``, or ``"build"``, matching the TTA pipeline stage in which the
failure occurred.  ``leaf_op`` identifies the annotated op and ``impl_id``
identifies the implementation (plugin name, builtin add_name, or lower_as key).
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
        Pipeline stage in which the error occurred.  One of ``"export"``
        (during ``torch.export``), ``"lowering"`` (during TRT converter /
        layer construction), or ``"build"`` (during TRT engine compilation).
    leaf_op:
        Qualified op name that triggered the error, e.g.
        ``"torch_tensorrt_anno_plugin::plugin_abc123"`` or the BuiltinSpec
        ``add_name`` / RegistryPluginSpec name.  May be ``None`` when not applicable.
    impl_id:
        Implementation identity string: plugin name/version, builtin
        ``add_name``, or custom_plugin description.  May be ``None`` when
        not applicable.

    Attributes
    ----------
    stage : str
        The pipeline stage label.
    leaf_op : str or None
        The leaf op identifier, if available.
    impl_id : str or None
        The implementation identifier, if available.
    message : str
        The raw human-readable message, without the structured prefix.
    """

    def __init__(
        self,
        msg: str,
        *,
        stage: str,
        leaf_op: Optional[str] = None,
        impl_id: Optional[str] = None,
    ) -> None:
        """Construct a TTADiagnosticError.

        Parameters
        ----------
        msg:
            Human-readable failure description.
        stage:
            TTA pipeline stage label (``"export"``, ``"lowering"``, or
            ``"build"``).
        leaf_op:
            Qualified op name or FX node name, or ``None``.
        impl_id:
            Implementation identity string, or ``None``.
        """
        self.stage = stage
        self.leaf_op = leaf_op
        self.impl_id = impl_id
        self.message = msg
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Build the structured prefix and combine it with the raw message.

        Subclasses may override this to customise the label used between the
        stage tag and the raw message text (e.g. ``(builtin=...)``) while
        still inheriting the base ``__str__`` implementation.

        Returns
        -------
        str
            Formatted error string of the form::

                [TTA <STAGE>] [<leaf_op>] [(impl=<impl_id>)]: <message>
        """
        parts: list[str] = [f"[TTA {self.stage.upper()}]"]
        if self.leaf_op:
            parts.append(self.leaf_op)
        if self.impl_id:
            parts.append(f"(impl={self.impl_id})")
        return f"{' '.join(parts)}: {self.message}"

    def __str__(self) -> str:
        """Return the full human-readable error string.

        The string includes the stage label, leaf op, implementation id, and
        the raw message, omitting any fields that are ``None``.

        Format::

            [TTA <STAGE>] <leaf_op> (impl=<impl_id>): <message>
        """
        return self._format_message()


class TTABuiltinError(TTADiagnosticError):
    """Error during TTA builtin lowering.

    Raised when a :class:`~torch_tensorrt.annotation._specs.BuiltinSpec`
    cannot be lowered to a TensorRT built-in layer.  Common causes:

    * The ``add_*`` method does not exist on the TRT ``INetworkDefinition``.
    * Argument binding fails (wrong types or unknown keyword arguments).
    * The TRT API rejects the arguments at layer-creation time.

    Parameters
    ----------
    msg:
        Human-readable failure description.
    stage:
        Pipeline stage; defaults to ``"lowering"``.
    leaf_op:
        FX node name or qualified op name that triggered the error, or
        ``None``.
    impl_id:
        The ``add_*`` method name (e.g. ``"add_elementwise"``), or ``None``.

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
        stage: str = "lowering",
        leaf_op: Optional[str] = None,
        impl_id: Optional[str] = None,
    ) -> None:
        """Construct a TTABuiltinError.

        Parameters
        ----------
        msg:
            Human-readable failure description.
        stage:
            TTA pipeline stage; defaults to ``"lowering"``.
        leaf_op:
            Qualified op name or FX node name, or ``None``.
        impl_id:
            The TRT ``add_*`` method name, or ``None``.
        """
        super().__init__(msg, stage=stage, leaf_op=leaf_op, impl_id=impl_id)

    def _format_message(self) -> str:
        """Build the structured message with a ``(builtin=...)`` label.

        Returns
        -------
        str
            Formatted string of the form::

                [TTA LOWERING] <leaf_op> (builtin=<impl_id>): <message>
        """
        parts: list[str] = [f"[TTA {self.stage.upper()}]"]
        if self.leaf_op:
            parts.append(self.leaf_op)
        if self.impl_id:
            parts.append(f"(builtin={self.impl_id})")
        return f"{' '.join(parts)}: {self.message}"


class TTAPluginError(TTADiagnosticError):
    """Error during TTA plugin lowering.

    Raised when a :class:`~torch_tensorrt.annotation._specs.RegistryPluginSpec`
    cannot be lowered via the TensorRT plugin registry.  Common causes:

    * The plugin is not registered in the TRT plugin registry.
    * Plugin field population fails (unsupported attribute type).
    * The TRT API rejects the plugin layer at creation time.

    Parameters
    ----------
    msg:
        Human-readable failure description.
    stage:
        Pipeline stage; defaults to ``"lowering"``.
    leaf_op:
        FX node name or qualified op name that triggered the error, or
        ``None``.
    impl_id:
        Plugin name/version string (e.g. ``"MyPlugin_TRT_v1"``), or ``None``.

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
        stage: str = "lowering",
        leaf_op: Optional[str] = None,
        impl_id: Optional[str] = None,
    ) -> None:
        """Construct a TTAPluginError.

        Parameters
        ----------
        msg:
            Human-readable failure description.
        stage:
            TTA pipeline stage; defaults to ``"lowering"``.
        leaf_op:
            Qualified op name or FX node name, or ``None``.
        impl_id:
            Plugin name/version string, or ``None``.
        """
        super().__init__(msg, stage=stage, leaf_op=leaf_op, impl_id=impl_id)

    def _format_message(self) -> str:
        """Build the structured message with a ``(plugin=...)`` label.

        Returns
        -------
        str
            Formatted string of the form::

                [TTA LOWERING] <leaf_op> (plugin=<impl_id>): <message>
        """
        parts: list[str] = [f"[TTA {self.stage.upper()}]"]
        if self.leaf_op:
            parts.append(self.leaf_op)
        if self.impl_id:
            parts.append(f"(plugin={self.impl_id})")
        return f"{' '.join(parts)}: {self.message}"


class LowerAsError(TTADiagnosticError):
    """Raised when the ``tta.lower_as`` FX pass cannot lower a region.

    Carries additional region-level context beyond the base
    :class:`TTADiagnosticError` fields.  Common causes:

    * The ``impl_id`` referenced by the annotation is not in the impl registry.
    * The region's IO signature is incompatible with the registered impl.
    * Op materialisation via :func:`get_or_create_op_for_boundary` fails.

    Parameters
    ----------
    msg:
        Human-readable failure description.
    rid:
        Integer region ID of the failing region, or ``None``.
    name:
        Human-readable region name (from the annotation), or ``None``.
    reason:
        Short code identifying the failure kind, e.g.
        ``"IO_MISMATCH"`` or ``"PLUGIN_NOT_FOUND"``.  This is also stored
        as ``impl_id`` on the base class.

    Attributes
    ----------
    stage : str
        Always ``"lowering"`` for this error type.
    leaf_op : str or None
        Derived as ``name`` when provided, else ``"region_<rid>"`` when
        ``rid`` is provided, else ``None``.
    impl_id : str or None
        Equal to ``reason``.
    rid : int or None
    name : str or None
    reason : str or None
    message : str
    """

    def __init__(
        self,
        msg: str,
        *,
        rid: Optional[int] = None,
        name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Construct a LowerAsError.

        Parameters
        ----------
        msg:
            Human-readable failure description.
        rid:
            Integer region ID, or ``None``.
        name:
            Human-readable region name from the annotation, or ``None``.
        reason:
            Short failure-kind code (also stored as ``impl_id``), or ``None``.
        """
        # Set subclass-specific attributes BEFORE calling super().__init__()
        # so that _format_message() (called inside super().__init__()) can
        # reference them safely.
        self.rid = rid
        self.name = name
        self.reason = reason
        leaf_op: Optional[str] = name or (
            f"region_{rid}" if rid is not None else None
        )
        super().__init__(msg, stage="lowering", leaf_op=leaf_op, impl_id=reason)

    def _format_message(self) -> str:
        """Build the structured message with region-specific labels.

        Returns
        -------
        str
            Formatted string of the form::

                [TTA LOWERING] <name|region_<rid>> (reason=<reason>): <msg>

            When ``rid`` is present but ``name`` is not, an explicit
            ``(rid=<rid>)`` label is appended for clarity.
        """
        parts: list[str] = [f"[TTA {self.stage.upper()}]"]
        if self.leaf_op:
            parts.append(self.leaf_op)
        if self.reason:
            parts.append(f"(reason={self.reason})")
        if self.rid is not None and not self.name:
            parts.append(f"(rid={self.rid})")
        return f"{' '.join(parts)}: {self.message}"
