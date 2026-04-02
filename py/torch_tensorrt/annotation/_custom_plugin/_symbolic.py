"""Symbolic tensor proxies used during AOT kernel compilation (shape expression bindings).

During the AOT compilation pipeline each backend (Triton, cuTILE, CuTe DSL) needs
to run the user's *launch function* without a real GPU so that it can:

  1. Intercept the kernel call and record which tensors / scalars are passed.
  2. Capture the symbolic launch grid (grid_x/y/z) as trtp.SymInt32 expressions
     so that TRT can evaluate them at engine-run time with actual input shapes.

``SymbolicTensor`` is the proxy object injected in place of real ``torch.Tensor``
arguments.  It wraps a QDP ``TensorDesc`` and exposes:

* ``shape`` / ``shape_expr`` — per-dimension *SymInt32 expressions* that bind to
  the input dimension at runtime (e.g. ``shape[0]`` evaluates to the batch size
  that TRT resolves during engine execution).
* ``stride`` — row-major symbolic strides, or format-specific strides for packed
  channel layouts (HWC, DHWC, etc.).
* ``numel()`` — symbolic product of all dimensions.

The ``TensorRole`` enum distinguishes input tensors from output tensors so that
``analyze_launch_args`` in ``_qdp_utils.py`` can reconstruct the correct
``param_binding_indices`` that TRT uses to identify which runtime buffer to pass
for each kernel pointer parameter.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
    trt = None  # type: ignore[assignment]
    trtp = None  # type: ignore[assignment]


class TensorRole(Enum):
    """Identifies whether a SymbolicTensor represents an input or output tensor.

    Used by ``analyze_launch_args`` to map each kernel pointer parameter back to
    the correct TRT binding index (inputs occupy indices 0..num_inputs-1, outputs
    occupy indices num_inputs..num_inputs+num_outputs-1).
    """

    INPUT = auto()
    OUTPUT = auto()


def _to_sym_int(x: Any) -> Any:
    """Convert a shape/stride element to SymInt32; accept int or SymInt32-like."""
    if _TRT_AVAILABLE and isinstance(x, trtp.SymInt32):
        return x
    if isinstance(x, int):
        return trtp.SymInt32(x) if _TRT_AVAILABLE else x
    if _TRT_AVAILABLE:
        return trtp.SymInt32(x)
    return x


def _sym_to_int_if_const(x: Any) -> Any:
    """Return Python int if x is concrete (Python int or constant SymInt32), else return x unchanged.

    In TRT's aot_impl context, shape_expr elements may be SymInt32 objects backed by
    a trt.IDimensionExpr constant (e.g. for a static-shape tensor).  We can extract
    the concrete value via _int_expr.is_constant() / .get_constant_value().
    This lets downstream code use isinstance(v, int) checks correctly so that grid
    computation in user launch_fn works even with SymbolicTensor arguments.
    """
    if isinstance(x, int):
        return x
    # Try via TRT IDimensionExpr (available when _exprBuilder was set during build).
    # Broad catch is necessary: IDimensionExpr attribute access raises different
    # exception types across TRT versions (AttributeError, RuntimeError, TypeError).
    try:
        ie = getattr(x, "_int_expr", None)
        if ie is not None and ie.is_constant():
            return int(ie.get_constant_value())
    except (AttributeError, RuntimeError, TypeError, ValueError) as _e:
        logger.debug("_sym_to_int_if_const: failed to extract constant from %r: %s", x, _e)
    return x


class _ShapeDim:
    """Wrapper for a symbolic shape dimension supporting arithmetic in launch fns.

    math.prod(iterable) starts with 1 (int) and multiplies left-to-right.
    The first step is ``1 * element``, which Python evaluates as:
      1. int.__mul__(element) → NotImplemented
      2. element.__rmul__(1)  ← this method

    Standard trtp.SymInt32 has no __rmul__, so math.prod fails for dynamic
    dims. _ShapeDim wraps a SymInt32 (or int) and delegates all arithmetic to
    self._v so that expressions like math.prod(x.shape) and ceiling division
    (M + BX - 1) // BX work transparently in @cute.jit launch functions.

    _expr is exposed so that SymIntExpr._op and _as_symint32 can extract the
    underlying IDimensionExpr without knowing the concrete wrapper type.
    """

    def __init__(self, v: Any) -> None:
        self._v = v  # int or trtp.SymInt32
        # Cache the underlying IDimensionExpr so SymIntExpr._op and _as_symint32
        # can extract it without knowing the concrete wrapper type.
        self._expr = getattr(v, "_expr", getattr(v, "_int_expr", None))

    def __add__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v + v

    def __radd__(self, other: Any) -> Any:
        return self._v + other

    def __sub__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v - v

    def __rsub__(self, other: Any) -> Any:
        return other - self._v

    def __mul__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v * v

    def __rmul__(self, other: Any) -> Any:
        """Called as: other * self (e.g., int(1) * _ShapeDim from math.prod)."""
        return self._v * other  # commutative: SymInt32.__mul__(int) handles it

    def __floordiv__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v // v

    def __rfloordiv__(self, other: Any) -> Any:
        return other // self._v

    def __int__(self) -> int:
        return int(self._v)

    def __repr__(self) -> str:
        return f"_ShapeDim({self._v!r})"


def _get_format(td: Any) -> Any:
    """Read the TRT format from a TensorDesc.

    td._format is set by TRT before aot_impl is called (has_shape=True at that
    point), so we try _format first (always accessible) then .format (requires
    has_shape=True as a guard).  Returns None when unavailable (e.g., during
    sandbox construction where has_shape=False).
    """
    if not _TRT_AVAILABLE:
        return None
    fmt = getattr(td, "_format", None)
    if fmt is not None:
        return fmt
    # Broad catch is necessary: .format raises different exception types across
    # TRT versions and binding states (AttributeError when has_shape is False
    # in some builds, RuntimeError in others).
    try:
        if getattr(td, "has_shape", False):
            return td.format
    except (AttributeError, RuntimeError) as _e:
        logger.debug("_get_format: failed to read .format from TensorDesc %r: %s", td, _e)
    return None


def _row_major_strides(shape_expr: Any) -> Tuple[Any, ...]:
    """Row-major (C-contiguous) strides from symbolic shape_expr."""
    prod = trtp.SymInt32(1) if _TRT_AVAILABLE else 1
    strides: list = []
    for i in range(len(shape_expr) - 1, -1, -1):
        strides.insert(0, prod)
        prod = prod * _to_sym_int(shape_expr[i])
    return tuple(_to_sym_int(s) for s in strides)


# LIMITATION: Packed channel formats (CHW2, CHW4, CHW16, CHW32, CDHW32) are not
# supported for symbolic stride computation.  These formats pack multiple channel
# values into a single storage element, so the byte offset of channel c is a
# piecewise-linear (not affine) function of c and cannot be expressed as a single
# TRT SymInt32 stride value per dimension.  _strides_from_format falls back to
# row-major strides with a warning for these formats.  If a kernel spec sets
# input_formats or output_formats to a packed channel format and TRT's autotuner
# selects it, the strides passed to the descriptor function will be incorrect.
def _strides_from_format(td: Any) -> Tuple[Any, ...]:
    """Compute symbolic strides from (shape_expr, format).

    Supported formats with simple per-dim stride formulas:

      LINEAR            — row-major for any ndim
      HWC (4D)          — physical [N,H,W,C]:        strides [H*W*C,  1, W*C,   C]
      HWC8 (4D)         — physical [N,H,W,R/8, 8]:   strides [H*W*R,  1, W*R,   R]
                          where R = roundup(C, 8) = ((C+7)//8)*8
      HWC16 (4D)        — physical [N,H,W,R/16, 16]: strides [H*W*R,  1, W*R,   R]
                          where R = roundup(C, 16)
      DHWC (5D)         — physical [N,D,H,W,C]:      strides [D*H*W*C, 1, H*W*C, W*C, C]
      DHWC8 (5D)        — physical [N,D,H,W,R/8, 8]: strides [D*H*W*R, 1, H*W*R, W*R, R]

    CHW2/4/16/32 and CDHW32 pack channels in the outermost dim; offset(c) is
    piecewise linear in c (not a single stride), so they fall back to row-major
    with a warning.
    """
    shape_expr = td.shape_expr
    ndim = len(shape_expr)
    fmt = _get_format(td)

    if fmt is None or fmt == trt.TensorFormat.LINEAR:
        return _row_major_strides(shape_expr)

    one = trtp.SymInt32(1) if _TRT_AVAILABLE else 1

    def sym(x: Any) -> Any:
        return _to_sym_int(x)

    if fmt == trt.TensorFormat.HWC and ndim == 4:
        # logical [N,C,H,W] — physical [N,H,W,C]
        _N, C, H, W = [sym(d) for d in shape_expr]
        WC = W * C
        return (H * WC, one, WC, C)

    if fmt in (trt.TensorFormat.HWC8, trt.TensorFormat.HWC16) and ndim == 4:
        pack = 8 if fmt == trt.TensorFormat.HWC8 else 16
        _N, C, H, W = [sym(d) for d in shape_expr]
        R = (C + sym(pack - 1)) // sym(pack) * sym(pack)  # roundup(C, pack)
        WR = W * R
        return (H * WR, one, WR, R)

    if fmt == trt.TensorFormat.DHWC and ndim == 5:
        # logical [N,C,D,H,W] — physical [N,D,H,W,C]
        _N, C, D, H, W = [sym(d) for d in shape_expr]
        WC = W * C
        HWC = H * WC
        return (D * HWC, one, HWC, WC, C)

    if fmt == trt.TensorFormat.DHWC8 and ndim == 5:
        _N, C, D, H, W = [sym(d) for d in shape_expr]
        R = (C + sym(7)) // sym(8) * sym(8)
        WR = W * R
        HWR = H * WR
        return (D * HWR, one, HWR, WR, R)

    warnings.warn(
        f"SymbolicTensor: format {fmt} uses packed channel groups that cannot be "
        f"expressed as simple per-dim strides; falling back to row-major. "
        f"Kernels that call .stride() with this format will compute wrong offsets.",
        stacklevel=4,
    )
    return _row_major_strides(shape_expr)


@dataclass
class SymbolicTensor:
    """Symbolic view over a QDP TensorDesc with role metadata.

    Attributes:
      td: TensorDesc from QDP.
      role: TensorRole.INPUT or TensorRole.OUTPUT.
      index: role-local index (0-based, within inputs or outputs).
    """

    td: Any  # trtp.TensorDesc
    role: TensorRole
    index: int

    def __post_init__(self) -> None:
        # Pre-compute shape dims once: Python int for static, _ShapeDim for dynamic.
        # _ShapeDim adds __rmul__ so math.prod(x.shape) works in @cute.jit kernels.
        # Guard against mock/test TDs that lack shape_expr.
        shape_expr = getattr(self.td, "shape_expr", None)
        _shape = []
        if shape_expr is not None:
            for d in shape_expr:
                concrete = _sym_to_int_if_const(d)
                _shape.append(concrete if isinstance(concrete, int) else _ShapeDim(concrete))
        self._shape: Tuple[Any, ...] = tuple(_shape)

        self._stride: Tuple[Any, ...] = _strides_from_format(self.td) if shape_expr is not None else ()

        # Pre-compute numel: Python int for fully-static shapes, SymInt32 for dynamic.
        if shape_expr is None:
            self._numel: Any = 0
        else:
            concrete_dims = [_sym_to_int_if_const(d) for d in shape_expr]
            if all(isinstance(v, int) for v in concrete_dims):
                result = 1
                for v in concrete_dims:
                    result *= v
                self._numel = result
            else:
                n = trtp.SymInt32(1) if _TRT_AVAILABLE else 1
                for d in shape_expr:
                    n = n * _to_sym_int(d)
                self._numel = n

    @property
    def shape(self) -> Tuple[Any, ...]:
        return self._shape

    @property
    def shape_expr(self) -> Tuple[Any, ...]:
        """Symbolic shape dimensions (same as .shape); use for grid or extra_args."""
        return self.shape

    def size(self, dim: int | None = None):
        """PyTorch-style alias for shape / shape[dim]."""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def shape_dim(self, dim: int) -> Any:
        """Return the raw SymInt32 for dimension *dim* from the underlying TensorDesc."""
        return _to_sym_int(self.td.shape_expr[dim])

    def stride(self, dim: int | None = None):
        """PyTorch-style stride API: stride() or stride(dim)."""
        if dim is None:
            return self._stride
        return self._stride[dim]

    def numel(self) -> Any:
        """Total element count: Python int for static shapes, SymInt32 for dynamic."""
        return self._numel

    @property
    def is_cuda(self) -> bool:
        return True


def cdiv(a: Any, b: int) -> Any:
    """Ceiling division on a SymInt32 by a Python int divisor.

    Only the pattern SymInt32 // int is required by the contract.
    """
    return (a + (b - 1)) // b
