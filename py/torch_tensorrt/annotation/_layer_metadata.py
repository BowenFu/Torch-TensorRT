"""TTA metadata stored on TensorRT ``ILayer`` objects as a plain string.

Role in the compilation pipeline
---------------------------------
This module is responsible for encoding and decoding the structured metadata
string that TTA attaches to every ``ILayer`` it creates.  Metadata is written
during the build phase via ``layer.metadata = <string>`` (TRT 10+) and becomes
visible in the TRT engine inspector when the builder verbosity is set to at
least ``ILogger.Severity.DETAILED``.

The metadata string lets post-build diagnostics and the engine inspector map
engine layers back to the original TTA spec, FX node, and PyTorch op path —
without requiring access to the Python session that performed the build.

Metadata format
---------------
All TTA-annotated layers use space-separated tokens starting with the sentinel
``tta``.  There are two tiers:

**Tier 1** — Backend-providing nodes (``lower_as`` / ``export_as`` /
``autotune`` / ``builtin``).  Full attribution including backend kind, plugin
name, compile-time constants, and the originating PyTorch op path::

    tta <backend>:<plugin_name> [fn:<fn_name>:<cfg>|...] attrs:<k>=<v>,... torch_op:<path>

The optional ``fn:`` token lists all kernel (fn_name, config) pairs, one per
(spec, config) combination, pipe-separated.  Each entry is
``<fn_name>:<k>=<v>,...`` (config keys sorted; empty string after the colon
when the config is empty).

Examples::

    tta triton:host_kernel_abc fn:launch_add_one:BLOCK_SIZE=128 attrs: torch_op:model.fc
    tta triton:host_kernel_def fn:launch_add_one:BLOCK_SIZE=64|launch_add_one:BLOCK_SIZE=128 attrs: torch_op:x
    tta builtin:add_activation attrs: torch_op:relu
    tta plugin:InstanceNormalization_TRT attrs:eps=1e-05 torch_op:norm_op

**Tier 2** — Non-backend region nodes (autocast, quantize, observe_perf, …)::

    tta torch_op:<path>

For ``observe_perf``, ``<path>`` is normally
``<nn_module_qname>/<fx_node.name>`` when ``torch.export`` populated
``nn_module_stack``, otherwise ``<fx_node.name>`` only.  Use
:func:`tta_observe_perf_torch_op_path` to build this string consistently for
both tag maps and layer metadata.

Parser contract
---------------
- Prefix ``tta`` identifies a TTA-annotated layer.
- Tokens are space-separated.
- ``<backend>:<plugin_name>`` — single colon separates backend from plugin name.
- ``attrs:<k>=<v>,...`` — comma-separated key=value pairs; empty string after
  the colon if there are no attrs (``attrs:``).
- ``torch_op:<path>`` — always the last token; path may contain dots, brackets,
  underscores, and forward slashes.
- No JSON, no nested braces — flat and unambiguous.

TRT verbosity note
------------------
Metadata is surfaced by the TRT engine inspector only when the builder is
configured with verbosity ``ILogger.Severity.DETAILED`` (or higher).  At lower
verbosity levels the field exists on the layer object during the build but will
not appear in the serialised plan inspector output.

Fused-layer handling
--------------------
TRT may fuse multiple annotated layers into one.  In that case the ``metadata``
field contains all original strings concatenated with the ASCII record separator
``\\x1f`` (0x1F).  :func:`parse_tta_layer_metadata` applies the
*innermost-region rule*: the Tier-1 backend annotation takes precedence over
region-wrapper annotations, which take precedence over Tier-2.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# torch_op path helpers
# ---------------------------------------------------------------------------


def tta_observe_perf_torch_op_path(node: Any) -> str:
    """Build the Tier-2 ``torch_op`` path for ``observe_perf`` metadata and tag keys.

    Prefers ``<nn_module_qname>/<fx_node.name>`` when ``torch.export``
    populated ``node.meta["nn_module_stack"]`` (same deepest-module rule used
    by TTA region views).  Falls back to ``node.name`` alone when the stack is
    absent or empty, preserving the pre-``nn_module_stack`` behaviour.

    Args:
        node: An ``fx.Node`` (or any object with ``name`` and ``meta``
              attributes).  Missing attributes are treated as empty values.

    Returns:
        A path string suitable for use in the ``torch_op:`` field of a TTA
        metadata string and as a key in ``observe_perf`` tag maps.
    """
    name: str = getattr(node, "name", "") or ""
    meta: Dict[str, Any] = getattr(node, "meta", None) or {}
    stack = meta.get("nn_module_stack")

    if isinstance(stack, dict) and stack:
        # Walk the stack in reverse insertion order to find the deepest module
        # with a non-empty qualified name.
        qname: str = ""
        for _key in reversed(list(stack.keys())):
            entry = stack[_key]
            if isinstance(entry, tuple) and len(entry) >= 1:
                q = entry[0]
                if q:
                    qname = str(q)
                    break
        if qname:
            return f"{qname}/{name}"

    return name


# ---------------------------------------------------------------------------
# Tactics string builder (autotune)
# ---------------------------------------------------------------------------


def tactics_string(specs: Sequence[Any]) -> str:
    """Build the tactics string encoding all (spec, config) combinations.

    Produces the compact ``idx:backend:launch_fn_name:config|...`` encoding
    stored in the ``autotune`` layer metadata and used by TRT tactic selection.

    Format details:

    - Entries are separated by ``|``.
    - ``idx`` is 1-based and matches the ``TacticValue`` assigned by TRT
      autotune.  Each (spec, config) pair gets a unique, monotonically
      increasing index across all specs in *specs*.
    - ``backend`` is one of ``"triton"``, ``"cutile"``, ``"cutedsl"``, or the
      lowercased class name for unknown spec types.
    - ``launch_fn_name`` is ``spec.launch_fn.__name__``, or ``"unknown"``
      when the attribute is absent.
    - ``config`` is a comma-separated ``key=value`` list sorted by key.
      Empty string when the config dict is empty.
    - When ``spec.configs`` is ``None`` or ``[]``, the spec contributes a
      single entry with an empty config string (every spec has at least one
      tactic).

    Args:
        specs: Sequence of spec objects (``TritonSpec``, ``CuTileSpec``,
               ``CuTeDSLSpec``, or any object with ``launch_fn`` and
               ``configs`` attributes).

    Returns:
        Pipe-separated tactics string, e.g.
        ``"1:triton:swiglu_kernel:BLOCK_M=64,BLOCK_N=128|2:triton:swiglu_kernel:BLOCK_M=128,BLOCK_N=64"``.
        Returns an empty string when *specs* is empty.
    """
    from ._specs import CuTeDSLSpec, CuTileSpec, TritonSpec

    parts: List[str] = []
    idx: int = 1
    for spec in specs:
        if isinstance(spec, TritonSpec):
            backend = "triton"
        elif isinstance(spec, CuTileSpec):
            backend = "cutile"
        elif isinstance(spec, CuTeDSLSpec):
            backend = "cutedsl"
        else:
            backend = type(spec).__name__.lower()

        fn_name: str = getattr(spec.launch_fn, "__name__", "unknown")
        # Fall back to a single empty config so every spec has at least one
        # tactic entry (required for 1-based TacticValue consistency).
        configs: List[Dict[str, Any]] = spec.configs if spec.configs else [{}]
        for cfg in configs:
            cfg_str: str = ",".join(f"{k}={v}" for k, v in sorted(cfg.items()))
            parts.append(f"{idx}:{backend}:{fn_name}:{cfg_str}")
            idx += 1

    return "|".join(parts)


# ---------------------------------------------------------------------------
# Metadata formatters
# ---------------------------------------------------------------------------


def _format_attrs(attrs: Optional[Dict[str, Any]]) -> str:
    """Serialise an attrs dict to a ``k=v,k=v,...`` string sorted by key.

    Returns an empty string when *attrs* is ``None``, empty, or falsy so that
    the ``attrs:`` token in the metadata string is present but empty
    (``attrs:``) rather than absent.

    Args:
        attrs: Dict of compile-time scalar constants (e.g. tiling parameters).

    Returns:
        Comma-separated ``"key=value"`` string sorted by key, or ``""``
        when *attrs* is empty or ``None``.
    """
    if not attrs:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(attrs.items()))


def _format_fn_specs(fn_specs: List[Tuple[str, Dict[str, Any]]]) -> str:
    """Encode a list of (fn_name, config) pairs as ``fn_name:k=v,...`` entries joined by ``|``."""
    parts = []
    for fn_name, cfg in fn_specs:
        cfg_str = ",".join(f"{k}={v}" for k, v in sorted(cfg.items()))
        parts.append(f"{fn_name}:{cfg_str}")
    return "|".join(parts)


def _parse_fn_specs(fn_str: str) -> List[Dict[str, Any]]:
    """Parse the value of a ``fn:`` token into a list of ``{fn_name, config}`` dicts."""
    result = []
    for entry in fn_str.split("|"):
        if not entry:
            continue
        fn_name, _, cfg_str = entry.partition(":")
        cfg: Dict[str, Any] = {}
        if cfg_str:
            for kv in cfg_str.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    cfg[k.strip()] = _parse_value(v.strip())
        result.append({"fn_name": fn_name, "config": cfg})
    return result


def _format_tta_metadata(
    backend: str,
    plugin_name: str,
    torch_op: str,
    attrs: Optional[Dict[str, Any]] = None,
    fn_specs: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> str:
    """Format a Tier-1 TTA metadata string.

    Args:
        backend:     Backend kind, e.g. ``"triton"``, ``"builtin"``,
                     ``"plugin"``.
        plugin_name: Plugin or op name within the backend.
        torch_op:    PyTorch op path, e.g. ``"model.encoder.layers[0].mlp"``.
        attrs:       Optional compile-time scalar constants.
        fn_specs:    Optional list of ``(fn_name, config)`` pairs — one entry
                     per (spec, config) combination for custom plugins.

    Returns:
        A Tier-1 metadata string.  When *fn_specs* is provided the ``fn:``
        token is inserted between the backend token and ``attrs:``.
    """
    attrs_str: str = _format_attrs(attrs)
    if fn_specs:
        fn_str = _format_fn_specs(fn_specs)
        return f"tta {backend}:{plugin_name} fn:{fn_str} attrs:{attrs_str} torch_op:{torch_op}"
    return f"tta {backend}:{plugin_name} attrs:{attrs_str} torch_op:{torch_op}"


def _format_tta_metadata_tier2(torch_op: str) -> str:
    """Format a Tier-2 TTA metadata string (``torch_op`` only).

    Used for region-wrapper layers (autocast, quantize, observe_perf, …) that
    do not have an explicit backend assignment.

    Args:
        torch_op: PyTorch op path or FX node name.

    Returns:
        A Tier-2 metadata string of the form ``"tta torch_op:<torch_op>"``.
    """
    return f"tta torch_op:{torch_op}"


# ---------------------------------------------------------------------------
# Metadata parsers
# ---------------------------------------------------------------------------


def _parse_value(v: str) -> Any:
    """Parse a metadata value string to ``int``, ``float``, or ``str``.

    Tries integer conversion first, then float, and falls back to the raw
    string.  This mirrors the serialisation done by :func:`_format_attrs`.

    Args:
        v: Raw value string from a ``key=value`` metadata token.

    Returns:
        ``int``, ``float``, or ``str`` depending on what the string represents.
    """
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_single_tta_segment(raw: str) -> Optional[Dict[str, Any]]:
    """Parse one TTA metadata segment (no ``\\x1f`` separators expected).

    Recognises both tiers:

    **Tier-1** result keys: ``"backend"`` (str), ``"plugin_name"`` (str),
    ``"attrs"`` (dict), ``"torch_op"`` (str).

    **Tier-2** result keys: ``"torch_op"`` (str).

    Returns ``None`` when the segment does not start with the ``tta``
    sentinel, is too short to be valid, or is structurally malformed (e.g.
    missing the colon in the backend token, or no ``attrs:`` prefix).

    Args:
        raw: A single un-split metadata segment string.

    Returns:
        Parsed dict on success, or ``None`` if *raw* is not a TTA segment.
    """
    tokens: List[str] = raw.strip().split()
    if not tokens or tokens[0] != "tta":
        return None

    # --- Tier 2: "tta torch_op:<path>" ---
    if len(tokens) == 2 and tokens[1].startswith("torch_op:"):
        return {"torch_op": tokens[1][len("torch_op:"):]}

    # --- Tier 1: "tta <backend>:<plugin_name> [fn:<names>] attrs:<k>=<v>,... torch_op:<path>" ---
    if len(tokens) < 4:
        # Fewer than 4 tokens cannot satisfy the Tier-1 format.
        return None

    backend_plugin: str = tokens[1]
    if ":" not in backend_plugin:
        return None
    backend, plugin_name = backend_plugin.split(":", 1)

    # Optional fn: token (present when fn_specs were provided at write time).
    tok_idx = 2
    fn_specs: List[Dict[str, Any]] = []
    if tokens[tok_idx].startswith("fn:"):
        fn_specs = _parse_fn_specs(tokens[tok_idx][len("fn:"):])
        tok_idx += 1

    if tok_idx >= len(tokens):
        return None
    attrs_token: str = tokens[tok_idx]
    if not attrs_token.startswith("attrs:"):
        return None
    attrs_str: str = attrs_token[len("attrs:"):]
    attrs: Dict[str, Any] = {}
    if attrs_str:
        for kv in attrs_str.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                attrs[k.strip()] = _parse_value(v.strip())

    tok_idx += 1
    if tok_idx >= len(tokens):
        return None
    torch_op_token: str = tokens[tok_idx]
    if not torch_op_token.startswith("torch_op:"):
        return None
    torch_op: str = torch_op_token[len("torch_op:"):]

    result: Dict[str, Any] = {
        "backend": backend,
        "plugin_name": plugin_name,
        "attrs": attrs,
        "torch_op": torch_op,
    }
    if fn_specs:
        result["fn_specs"] = fn_specs
    return result


# ---------------------------------------------------------------------------
# Segment priority for fused-layer disambiguation
# ---------------------------------------------------------------------------

# Backends that represent a specific op lowering (higher priority than
# region wrappers such as autocast/observe_perf/autotune).
_TIER1_BACKENDS: frozenset = frozenset({"builtin", "plugin", "triton", "cutile", "cutedsl"})


def _segment_priority(parsed: Optional[Dict[str, Any]]) -> int:
    """Return the disambiguation priority for a parsed TTA metadata segment.

    When TRT fuses multiple annotated layers, their metadata strings are
    concatenated with ``\\x1f``.  The caller uses this priority to pick the
    *most specific* segment (innermost-region rule).

    Priority levels:

    - ``2`` — Tier-1 backend (builtin / plugin / triton / cutile / cutedsl)
    - ``1`` — Other Tier-1 segment (observe_perf, autotune, …)
    - ``0`` — Tier-2 segment (``torch_op`` only)
    - ``-1`` — Not a TTA segment (``parsed`` is ``None``)

    Args:
        parsed: Result of :func:`_parse_single_tta_segment`, or ``None``.

    Returns:
        Integer priority value in ``{-1, 0, 1, 2}``.
    """
    if parsed is None:
        return -1
    if "backend" in parsed:
        return 2 if parsed["backend"] in _TIER1_BACKENDS else 1
    return 0  # Tier-2


def parse_tta_layer_metadata(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a TTA metadata string (possibly fused) into a dict.

    TRT may fuse multiple annotated layers and concatenate their metadata
    strings with ``\\x1f`` (ASCII record separator, 0x1F).  This function
    applies the *innermost-region rule*: the highest-priority segment wins.
    Tier-1 backend annotations beat region-wrapper annotations (observe_perf,
    autotune, …), which beat Tier-2.

    **Tier-1** result keys: ``"backend"`` (str), ``"plugin_name"`` (str),
    ``"attrs"`` (dict), ``"torch_op"`` (str).

    **Tier-2** result keys: ``"torch_op"`` (str).

    Args:
        raw: Raw ``layer.metadata`` string from the TRT engine inspector.
             May contain ``\\x1f``-separated segments.

    Returns:
        The highest-priority parsed segment, or ``None`` if no segment starts
        with the ``tta`` sentinel (i.e. the layer was not annotated by TTA).
    """
    if not raw or not raw.strip():
        return None

    segments: List[str] = raw.split("\x1f")
    best: Optional[Dict[str, Any]] = None
    best_pri: int = -1
    for seg in segments:
        parsed = _parse_single_tta_segment(seg)
        pri = _segment_priority(parsed)
        if pri > best_pri:
            best, best_pri = parsed, pri
    return best


# ---------------------------------------------------------------------------
# Layer metadata writer
# ---------------------------------------------------------------------------


def set_tta_layer_metadata(
    layer: Any,
    backend: str,
    plugin_name: str,
    torch_op: str,
    attrs: Optional[Dict[str, Any]] = None,
    fn_specs: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> None:
    """Write a Tier-1 TTA metadata string to a TRT ``ILayer``.

    Tries both the ``metadata`` property (TRT 10+) and the ``set_metadata``
    method for forward/backward compatibility.  A read-only property error is
    logged as a warning and suppressed (non-fatal) so that builds on older TRT
    versions can continue without metadata.  All other exceptions are re-raised.

    ``attrs`` should contain only scalar compile-time constants such as tiling
    parameters or value constants.  Tensor weight bindings must **not** be
    included here.

    Args:
        layer:       The TRT ``ILayer`` to annotate.
        backend:     Backend kind, e.g. ``"triton"``, ``"builtin"``, ``"plugin"``.
        plugin_name: Plugin or op name within the backend namespace.
        torch_op:    PyTorch op path or FX node name for inspector attribution.
        attrs:       Optional dict of compile-time scalar constants.  Defaults
                     to ``None`` (empty attrs in the serialised string).

    Raises:
        Any exception from ``layer.metadata`` or ``layer.set_metadata()``
        other than ``AttributeError`` (which is treated as a non-fatal
        read-only-property condition on older TRT versions).
    """
    meta: str = _format_tta_metadata(backend, plugin_name, torch_op, attrs, fn_specs)
    try:
        if hasattr(layer, "metadata"):
            layer.metadata = meta
        elif hasattr(layer, "set_metadata"):
            layer.set_metadata(meta)
    except AttributeError as e:
        # Read-only property on this TRT version — non-fatal, log and continue
        # so that builds on older TRT versions are not broken by missing
        # metadata support.
        logger.warning(
            "Could not set TTA metadata on layer '%s' (AttributeError: %s). "
            "Metadata will be absent for this layer in the engine inspector.",
            getattr(layer, "name", "<unknown>"),
            e,
        )
