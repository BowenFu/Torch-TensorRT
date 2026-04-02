"""
TTA SignatureModel — TensorRT INetworkDefinition parameter introspection.

Role in the compilation pipeline
----------------------------------
When a user annotates a subgraph with ``tta.builtin(trt.add_convolution_nd, ...)``,
the annotation layer must later lower that subgraph into an actual TensorRT
``INetworkDefinition`` call.  To do that correctly it needs to know:

1. Which positional parameters of ``add_*`` are *tensor* inputs (``ITensor``)
   that must be wired from the TRT network at lowering time, vs.
2. Which parameters are *configuration* values (``int``, ``trt.Dims``,
   ``trt.Weights``, …) supplied by the user in the annotation kwargs, and
3. Which configuration parameters are required (no default) vs. optional.

``SignatureModel`` holds exactly this information for a single ``add_*``
method.  It is built once per method name by ``_extract_from_trt_doc``, which
inspects ``trt.INetworkDefinition.<add_name>.__doc__`` at import time, and is
then cached in ``_CACHE`` for subsequent calls.

The ``_binder`` module consumes ``SignatureModel`` to partition user-supplied
kwargs into constructor arguments and post-construction ``setattr`` calls.

Why docstring parsing instead of ``inspect.signature``?
--------------------------------------------------------
TensorRT's Python bindings are pybind11 extensions; ``inspect.signature``
cannot recover the parameter list from them.  The only structured source of
parameter information is the C++-generated pybind11 docstring, so we parse
that instead.  See ``_parse_params_from_doc`` for the exact format we rely on.
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import tensorrt as trt  # type: ignore[import]

from ._specs import BuiltinSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _ParamInfo:
    """Information about a single parameter extracted from a TRT docstring."""

    name: str
    has_default: bool
    is_tensor: bool  # True when the pybind11 type annotation contains ITensor or ILayer
    type_str: str = ""  # Raw type annotation string copied verbatim from the docstring


@dataclass(frozen=True)
class SignatureModel:
    """Parsed representation of one ``trt.INetworkDefinition.add_*`` method.

    Instances are immutable and cached; obtain them via
    :func:`get_signature_model` rather than constructing directly.

    Attributes:
        add_name: Name of the INetworkDefinition method, e.g.
            ``"add_convolution_nd"``.
        ctor_param_names: Ordered tuple of all non-self parameter names as
            they appear in the method signature.  Used to map kwargs to
            positional slots when calling the method.
        required_params: Set of parameter names that have no default value and
            are not tensor inputs.  These are mandatory configuration args
            (e.g. ``num_output_maps``, ``kernel_shape``) that the user *must*
            supply in the annotation kwargs.
        tensor_params: Set of parameter names whose pybind11 type annotation
            contains ``ITensor`` or ``ILayer``.  These are wired from the live
            TRT network at lowering time and must *not* appear in user kwargs.
        param_types: Mapping from parameter name to the raw type-annotation
            string extracted from the docstring (e.g.
            ``"tensorrt.tensorrt.ITensor"``).  Used by the binder for
            value-conversion heuristics.
    """

    add_name: str
    ctor_param_names: Tuple[str, ...]
    required_params: Set[str]
    tensor_params: Set[str]
    param_types: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# Module-level cache — keyed by add_name
# ---------------------------------------------------------------------------

_CACHE: Dict[str, SignatureModel] = {}
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Docstring parsing helpers
# ---------------------------------------------------------------------------


def _parse_params_from_doc(doc: str, add_name: str) -> List[_ParamInfo]:
    """Parse parameter information from a TensorRT pybind11 method docstring.

    TRT docstring format (pybind11-generated)
    -----------------------------------------
    pybind11 emits one or more overload signatures at the *top* of the
    docstring, each looking like::

        add_convolution_nd(self: tensorrt.tensorrt.INetworkDefinition,
                           input: tensorrt.tensorrt.ITensor,
                           num_output_maps: int,
                           kernel_shape: tensorrt.tensorrt.Dims,
                           kernel: tensorrt.tensorrt.Weights,
                           bias: tensorrt.tensorrt.Weights = None) -> IConvolutionLayer

    Key properties we rely on (and that make this parsing fragile):

    * The method name appears verbatim at the start of the first line.
    * Each parameter is ``name: TypeAnnotation`` or ``name: Type = default``.
    * ``self`` is always the first parameter and is skipped.
    * Parameters may span multiple lines (long signatures are line-wrapped).
    * The return type follows ``->`` after the closing ``)`` on the same or a
      continuation line — we stop collecting at the matching ``)``.
    * Type annotations use the fully-qualified pybind11 module path, e.g.
      ``tensorrt.tensorrt.ITensor`` rather than just ``ITensor``.
    * Methods with multiple overloads repeat the signature block; we only
      parse the first one.

    If pybind11 or TRT changes its docstring format, this parser will
    silently return an empty list (or mis-parse parameters).  The caller
    (``_extract_from_trt_doc``) will then raise a ``ValueError`` with a
    description of what it found, making the breakage visible.

    Args:
        doc: Raw docstring text (as returned by ``inspect.getdoc``).
        add_name: The method name to anchor the search (e.g.
            ``"add_convolution_nd"``).

    Returns:
        List of :class:`_ParamInfo` objects, one per non-self parameter, in
        declaration order.  Returns an empty list if parsing fails.
    """
    if not doc:
        return []

    lines = doc.strip().splitlines()

    # Locate the line that starts the signature.  We prefer a line that
    # contains add_name followed by ``(``, which uniquely identifies the
    # pybind11 overload header.
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if add_name in line and "(" in line:
            start_idx = i
            break
    if start_idx is None:
        # Fallback: any line with an opening paren (handles unusual formatting).
        for i, line in enumerate(lines):
            if "(" in line:
                start_idx = i
                break
    if start_idx is None:
        return []

    # Trim the anchor line so it starts at add_name (strips any leading
    # whitespace or decoration added by pybind11).
    anchor_line = lines[start_idx]
    if add_name in anchor_line:
        anchor_line = anchor_line[anchor_line.find(add_name):]

    lpar = anchor_line.find("(")
    if lpar == -1:
        return []

    # Walk character-by-character from after the opening '(' to find the
    # matching ')'.  We track nesting depth so that parentheses inside
    # default-value expressions (rare but possible) are handled correctly.
    collected = anchor_line[lpar + 1:]  # text after the opening '('
    depth = 1
    inner_parts: List[str] = []
    for ch in collected:
        if ch == "(":
            depth += 1
            inner_parts.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
            inner_parts.append(ch)
        else:
            inner_parts.append(ch)

    if depth != 0:
        # The signature spans multiple lines — keep reading until depth == 0.
        for line in lines[start_idx + 1:]:
            for ch in line.strip():
                if ch == "(":
                    depth += 1
                    inner_parts.append(ch)
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                    inner_parts.append(ch)
                else:
                    inner_parts.append(ch)
            if depth == 0:
                break
            inner_parts.append(" ")  # collapsed line boundary

    if depth != 0:
        # Parenthesis never closed — malformed signature; bail out.
        logger.debug(
            "_parse_params_from_doc: unclosed '(' in docstring for %s; "
            "returning empty param list",
            add_name,
        )
        return []

    inner = "".join(inner_parts)
    if not inner.strip():
        return []

    params: List[_ParamInfo] = []
    for raw in inner.split(","):
        token = raw.strip()
        if not token:
            continue

        # Detect and strip default value.  pybind11 writes ``= None`` or
        # ``= value`` after the type annotation.
        has_default = "=" in token
        if has_default:
            token = token.split("=", 1)[0].strip()

        # Split ``name: TypeAnnotation`` into name and type parts.
        type_str = ""
        if ":" in token:
            name_part, type_str = token.split(":", 1)
            name_part = name_part.strip()
            type_str = type_str.strip()
        else:
            name_part = token
            type_str = ""

        # The name may have leading qualifiers in unusual cases; take the last
        # whitespace-delimited token.
        name_tokens = name_part.split()
        if not name_tokens:
            continue
        name = name_tokens[-1]

        # Skip receiver parameters — pybind11 always emits ``self`` first.
        if name in ("self", "this", "cls") or not name:
            continue

        # Tensor inputs are identified by their pybind11 type annotation
        # containing ``ITensor`` or ``ILayer``.  All other params are config.
        is_tensor = "ITensor" in type_str or "ILayer" in type_str

        params.append(
            _ParamInfo(
                name=name,
                has_default=has_default,
                is_tensor=is_tensor,
                type_str=type_str,
            )
        )

    return params


def _parse_param_names_from_doc(doc: str, add_name: str) -> Tuple[str, ...]:
    """Return an ordered tuple of parameter names from a TRT method docstring.

    This is a thin backward-compatibility wrapper around
    :func:`_parse_params_from_doc` for callers that only need the names.

    Args:
        doc: Raw docstring text.
        add_name: Method name used to anchor the signature search.

    Returns:
        Tuple of parameter names in declaration order, excluding ``self``.
    """
    params = _parse_params_from_doc(doc, add_name)
    return tuple(p.name for p in params)


def _available_add_methods() -> List[str]:
    """Return sorted list of ``add_*`` method names on ``trt.INetworkDefinition``.

    Used to populate error messages when an unknown method is requested.
    """
    return sorted(
        name
        for name in dir(trt.INetworkDefinition)
        if name.startswith("add_") and callable(getattr(trt.INetworkDefinition, name, None))
    )


def _extract_from_trt_doc(add_name: str) -> SignatureModel:
    """Build a :class:`SignatureModel` by inspecting ``trt.INetworkDefinition.<add_name>``.

    Looks up *add_name* on :class:`tensorrt.INetworkDefinition`, retrieves its
    pybind11-generated docstring, and delegates to :func:`_parse_params_from_doc`
    to extract parameter information.

    Args:
        add_name: Name of the method to introspect, e.g. ``"add_convolution_nd"``.

    Returns:
        A fully populated :class:`SignatureModel`.

    Raises:
        ValueError: If *add_name* does not exist on ``INetworkDefinition``, if
            the method has no docstring, or if the docstring yields no
            parameters (which would indicate a format change in pybind11 output).
    """
    method = getattr(trt.INetworkDefinition, add_name, None)
    if method is None:
        available = _available_add_methods()
        raise ValueError(
            f"Method '{add_name}' not found on trt.INetworkDefinition. "
            f"Available add_* methods ({len(available)}): {available}"
        )

    doc = inspect.getdoc(method)
    if not doc:
        raise ValueError(
            f"trt.INetworkDefinition.{add_name} has no docstring. "
            "This may indicate a TensorRT version that exposes the method "
            "without a pybind11 signature, which SignatureModel cannot parse."
        )

    params = _parse_params_from_doc(doc, add_name)

    if not params:
        # Surface the raw docstring so the maintainer can diagnose the mismatch.
        raise ValueError(
            f"Failed to parse any parameters from the docstring of "
            f"trt.INetworkDefinition.{add_name}. "
            "The pybind11 docstring format may have changed in this TRT version. "
            f"Raw docstring (first 400 chars):\n{doc[:400]!r}"
        )

    ctor_param_names = tuple(p.name for p in params)

    # Tensor params (ITensor/ILayer) are wired at lowering time from the live
    # TRT network; they must not appear in user-supplied kwargs.
    tensor_params = {p.name for p in params if p.is_tensor}

    # Required params have no default and are not tensor inputs.  The user
    # must supply these in the annotation kwargs (e.g. num_output_maps).
    required_params = {
        p.name for p in params
        if not p.has_default and not p.is_tensor
    }

    param_types = {p.name: p.type_str for p in params if p.type_str}

    return SignatureModel(
        add_name=add_name,
        ctor_param_names=ctor_param_names,
        required_params=required_params,
        tensor_params=tensor_params,
        param_types=param_types,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_signature_model(spec_or_name: Union[BuiltinSpec, str]) -> SignatureModel:
    """Return the cached :class:`SignatureModel` for an ``add_*`` method.

    Accepts either a :class:`~._specs.BuiltinSpec` (from which
    ``spec.add_name`` is read) or a bare string such as
    ``"add_convolution_nd"``.

    The first call for a given *add_name* builds the model by parsing the
    pybind11 docstring of ``trt.INetworkDefinition.<add_name>`` and stores it
    in a process-wide cache.  Subsequent calls return the cached instance
    without re-parsing.  Cache population is protected by ``_CACHE_LOCK`` so
    concurrent callers are safe.

    Args:
        spec_or_name: A :class:`~._specs.BuiltinSpec` or a string naming the
            ``INetworkDefinition`` method to introspect.

    Returns:
        The :class:`SignatureModel` for the requested method.

    Raises:
        ValueError: If the method does not exist on ``INetworkDefinition``, has
            no parseable docstring, or yields no parameters.
    """
    if isinstance(spec_or_name, str):
        add_name = spec_or_name
    else:
        add_name = spec_or_name.add_name

    with _CACHE_LOCK:
        if add_name in _CACHE:
            return _CACHE[add_name]

    model = _extract_from_trt_doc(add_name)

    with _CACHE_LOCK:
        # Double-checked locking: another thread may have populated the cache
        # while we were parsing; prefer the already-stored value to avoid
        # returning a different (but equivalent) object.
        if add_name not in _CACHE:
            _CACHE[add_name] = model
        return _CACHE[add_name]
