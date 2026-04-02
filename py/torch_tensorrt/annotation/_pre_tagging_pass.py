"""Pre-tagging compile pass for the TTA annotation layer.

What is "pre-tagging"?
-----------------------
Pre-tagging is the act of stamping TRTInterpreter-readable metadata directly
onto FX graph nodes *before* any rewrite pass (lower_as, autotune) or the
interpreter itself runs.  The metadata is stored in ``n.meta``, the standard
per-node side-channel used throughout torch.fx (e.g. ``nn_module_stack``).

What gets tagged?
-----------------
Two categories of nodes receive additional ``n.meta`` keys:

``n.meta["mark_debug"] = True``
    Set on nodes whose TRT output tensors should be marked via
    ``INetworkDefinition.mark_debug()``.  Once marked, TensorRT forwards the
    corresponding tensor values to any registered ``IDebugListener`` at
    inference time, enabling the ``tta.debug.expect`` capture mechanism.

``n.meta["layer_metadata"] = "tta torch_op:<path>"``
    Set on nodes that belong to an ``tta.observe_perf`` annotated region.
    ``TRTInterpreter`` reads this key and stamps it on the resulting TRT
    layer's ``metadata`` field.  The ``parse_tta_layer_metadata`` helper in
    ``_layer_metadata.py`` can then identify these layers by name during
    profiling attribution.  Nodes *without* this key receive a generic
    ``"{path} [{op}]"`` label from the interpreter.

When does this pass run?
------------------------
``apply_pre_tagging`` is registered as the *first* compile pass in the TTA
pipeline (see ``_compile/``).  It runs immediately after the exported program
is prepared, before:

1. ``lower_as`` rewrites that replace annotated subgraphs with TTA leaf ops.
2. ``autotune`` rewrites that splice in alternative op implementations.
3. ``TRTInterpreter`` which consumes the final FX graph.

Why must it run before autotune rewrites?
-----------------------------------------
Autotune may swap a node for a different op overload that carries none of the
original node's annotations.  The pre-tagging pass reads the node *names*
recorded by ``export_as`` / ``tta.observe_perf`` at Python annotation time
and stamps them while the FX graph still contains nodes with those exact
names.  Running after autotune would silently miss nodes that were renamed or
replaced.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


def apply_pre_tagging(
    exported_program: Any,
    fx_module: Any,
    logger: Optional[logging.Logger] = None,
    **_kwargs: Any,
) -> None:
    """Annotate FX nodes with TRTInterpreter hints drawn from ``ep._tta``.

    Reads the ``_tta`` side-dict attached to *exported_program* and stamps
    the relevant metadata keys onto matching FX nodes in *fx_module*.

    This function is intentionally a side-effecting pass that returns
    ``None``; it communicates results through ``n.meta``, not through its
    return value, so that it can be composed with other compile passes
    without requiring any changes to the pass-runner contract.

    Args:
        exported_program: The ``torch.export.ExportedProgram`` produced by
            ``torch.export.export`` or ``tta.compile``.  Must expose a
            ``._tta`` dict attribute (populated by the annotation layer);
            if the attribute is absent or ``None`` the function is a no-op.
        fx_module: The ``torch.fx.GraphModule`` whose nodes are to be
            tagged.  Modified in-place via ``n.meta``.
        logger: Optional pre-configured :class:`logging.Logger` to use for
            debug output.  Falls back to this module's logger when ``None``.
        **_kwargs: Ignored.  Present so the function can be called by a
            generic pass-runner that forwards extra keyword arguments.
    """
    _log: logging.Logger = logger or logging.getLogger(__name__)
    tta_meta: Dict[str, Any] = getattr(exported_program, "_tta", None) or {}

    # --- mark_debug: set n.meta["mark_debug"] = True on debug nodes ---
    debug_names: Set[str] = set(tta_meta.get("debug_node_names") or [])
    if debug_names:
        count: int = 0
        for n in fx_module.graph.nodes:
            if n.name in debug_names:
                n.meta["mark_debug"] = True
                count += 1
        if count:
            _log.debug("TTA pre-tagging: %d nodes marked for debug", count)

    # --- layer_metadata: TTA Tier-1 for export_as / lower_as TTA leaf ops ---
    # After run_decompositions() all node.meta is lost (new node objects).
    # Restore layer_metadata for every TTA leaf op node using the
    # op_target → name map stored in ep._tta, which survives decompositions.
    _op_name_map: Dict[str, str] = tta_meta.get("lower_as_op_to_name") or {}
    if _op_name_map:
        try:
            import torch._ops as _torch_ops

            count = 0
            for n in fx_module.graph.nodes:
                if (
                    n.op == "call_function"
                    and isinstance(n.target, _torch_ops.OpOverload)
                    and n.target.namespace.startswith("torch_tensorrt_anno_")
                    and not n.meta.get("layer_metadata")
                ):
                    _op_name = _op_name_map.get(str(n.target))
                    if _op_name:
                        n.meta["layer_metadata"] = _op_name
                        count += 1
            if count:
                _log.debug(
                    "TTA pre-tagging: %d lower_as layer_metadata entries restored",
                    count,
                )
        except Exception:
            pass

    # --- layer_metadata: TTA Tier-2 for observe_perf nodes ---
    # All other layers get a generic label from TRTInterpreter; observe_perf
    # nodes need the "tta torch_op:..." format so parse_tta_layer_metadata
    # can find them for region attribution.
    observe_perf_tags: Dict[str, str] = tta_meta.get("observe_perf_tags") or {}
    if observe_perf_tags:
        from ._layer_metadata import tta_observe_perf_torch_op_path

        count = 0
        for n in fx_module.graph.nodes:
            # Skip TTA annotation leaf ops (export_as / lower_as boundary ops).
            # Their Tier-1 metadata is written by the TRT converter directly via
            # set_tta_layer_metadata, which reads node.meta["layer_metadata"] as
            # the torch_op value.  Setting "tta torch_op:..." here would corrupt
            # that value (the converter would embed the full formatted string as
            # torch_op, which contains spaces and confuses the parser).
            if (
                n.op == "call_function"
                and hasattr(n.target, "namespace")
                and (n.target.namespace or "").startswith("torch_tensorrt_anno_")
            ):
                continue
            op_path: str = tta_observe_perf_torch_op_path(n)
            region_name: Optional[str] = (
                observe_perf_tags.get(op_path) or observe_perf_tags.get(n.name)
            )
            if region_name:
                n.meta["layer_metadata"] = f"tta torch_op:{op_path}"
                count += 1
        if count:
            _log.debug(
                "TTA pre-tagging: %d observe_perf layer_metadata entries set", count
            )
