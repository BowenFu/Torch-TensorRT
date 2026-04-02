"""
CuTe DSL kernels with parallel grids for E2E testing.

Each kernel uses cute.arch.block_idx() so the launch grid depends on tensor
shape — matching the CuTile and Triton test patterns.

Requires cutlass.cute (CUTLASS Python CuTe DSL).
"""

from __future__ import annotations

import math

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ── 1D kernels (grid = (N, 1, 1) — one block per element) ──────────────────
#
# math.prod(x.shape) gives the total element count for any tensor shape.
# This is the natural CuTeDSL pattern — no TTA-specific helper is needed.

@cute.kernel
def _add_one_kern(x: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    out[idx] = x[idx] + x.element_type(1.0)


@cute.jit
def launch_add_one(x: cute.Tensor, out: cute.Tensor):
    _add_one_kern(x, out).launch(grid=(math.prod(x.shape), 1, 1), block=(1, 1, 1))


@cute.kernel
def _scale_kern(x: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    out[idx] = x[idx] * x.element_type(2.0)


@cute.jit
def launch_scale(x: cute.Tensor, out: cute.Tensor):
    _scale_kern(x, out).launch(grid=(math.prod(x.shape), 1, 1), block=(1, 1, 1))


@cute.kernel
def _fma_kern(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    out[idx] = a[idx] * b[idx] + c[idx]


@cute.jit
def launch_fma(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, out: cute.Tensor):
    _fma_kern(a, b, c, out).launch(grid=(math.prod(a.shape), 1, 1), block=(1, 1, 1))


@cute.kernel
def _fused_add_relu_kern(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    val = x[idx] + y[idx]
    out[idx] = cute.arch.fmax(val, x.element_type(0.0))


@cute.jit
def launch_fused_add_relu(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor):
    _fused_add_relu_kern(x, y, out).launch(grid=(math.prod(x.shape), 1, 1), block=(1, 1, 1))


@cute.kernel
def _gelu_kern(x: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    v = x[idx]
    c = cute.Float32(0.7978845608028654)
    inner = c * (v + cute.Float32(0.044715) * v * v * v)
    t = cute.tanh(inner)
    out[idx] = cute.Float32(0.5) * v * (cute.Float32(1.0) + t)


@cute.jit
def launch_gelu(x: cute.Tensor, out: cute.Tensor):
    _gelu_kern(x, out).launch(grid=(math.prod(x.shape), 1, 1), block=(1, 1, 1))


# ── 2D kernel (grid = (M, N, 1) — one block per element) ───────────────────

@cute.kernel
def _add_2d_kern(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor):
    m = cute.arch.block_idx()[0]
    n = cute.arch.block_idx()[1]
    out[m, n] = x[m, n] + y[m, n]


@cute.jit
def launch_add_2d(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor):
    M = x.shape[0]
    N = x.shape[1]
    _add_2d_kern(x, y, out).launch(grid=(M, N, 1), block=(1, 1, 1))


# ── Matmul (grid = (M, N, 1) — one block per output element) ───────────────

@cute.kernel
def _matmul_kern(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    m = cute.arch.block_idx()[0]
    n = cute.arch.block_idx()[1]
    K = a.shape[1]
    acc = a.element_type(0.0)
    for k in range(K):
        acc = acc + a[m, k] * b[k, n]
    out[m, n] = acc


@cute.jit
def launch_matmul(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    M = a.shape[0]
    N = b.shape[1]
    _matmul_kern(a, b, out).launch(grid=(M, N, 1), block=(1, 1, 1))


# ── Conv1x1 + ReLU (grid = (N*H*W, 1, 1) — one block per spatial position) ─

@cute.kernel
def _conv1x1_relu_kern(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
):
    pid = cute.arch.block_idx()[0]
    H = x.shape[2]
    W = x.shape[3]
    C_in = x.shape[1]
    C_out = weight.shape[0]
    n = pid // (H * W)
    rest = pid % (H * W)
    h = rest // W
    w = rest % W
    for c_out in range(C_out):
        acc = x.element_type(0.0)
        for c_in in range(C_in):
            acc = acc + x[n, c_in, h, w] * weight[c_out, c_in]
        acc = acc + bias[c_out]
        out[n, c_out, h, w] = cute.arch.fmax(acc, x.element_type(0.0))


@cute.jit
def launch_conv1x1_relu(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
):
    N = x.shape[0]
    H = x.shape[2]
    W = x.shape[3]
    _conv1x1_relu_kern(x, weight, bias, out).launch(
        grid=(N * H * W, 1, 1), block=(1, 1, 1)
    )


# ── int32 kernel (grid = (N, 1, 1) — out = x + 1 via integer arithmetic) ──

@cute.kernel
def _add_one_int_kern(x: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    out[idx] = x[idx] + x.element_type(1)


@cute.jit
def launch_add_one_int(x: cute.Tensor, out: cute.Tensor):
    _add_one_int_kern(x, out).launch(grid=(math.prod(x.shape), 1, 1), block=(1, 1, 1))


# ── Blocked kernels (block > 1, grid = ceil(N / BLOCK_SIZE)) ───────────────
#
# block_id * block_size + thread_id gives a unique 1D index for each thread.
# BLOCK_SIZE is a config param so TRT autotune can select among tile sizes.

@cute.kernel
def _add_one_blocked_kern(x: cute.Tensor, out: cute.Tensor):
    block_id = cute.arch.block_idx()[0]
    thread_id = cute.arch.thread_idx()[0]
    block_size = cute.arch.block_dim()[0]
    idx = block_id * block_size + thread_id
    out[idx] = x[idx] + x.element_type(1.0)


@cute.jit
def launch_add_one_blocked(x: cute.Tensor, out: cute.Tensor, BLOCK_SIZE: int = 128):
    N = math.prod(x.shape)
    n_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    _add_one_blocked_kern(x, out).launch(grid=(n_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


@cute.kernel
def _scale_blocked_kern(x: cute.Tensor, out: cute.Tensor):
    block_id = cute.arch.block_idx()[0]
    thread_id = cute.arch.thread_idx()[0]
    block_size = cute.arch.block_dim()[0]
    idx = block_id * block_size + thread_id
    out[idx] = x[idx] * x.element_type(2.0)


@cute.jit
def launch_scale_blocked(x: cute.Tensor, out: cute.Tensor, BLOCK_SIZE: int = 128):
    N = math.prod(x.shape)
    n_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    _scale_blocked_kern(x, out).launch(grid=(n_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


# ── 2D blocked kernel (grid = (ceil(M/BX), ceil(N/BY), 1)) ─────────────────

@cute.kernel
def _add_one_2d_blocked_kern(x: cute.Tensor, out: cute.Tensor):
    bx = cute.arch.block_idx()[0]
    by = cute.arch.block_idx()[1]
    tx = cute.arch.thread_idx()[0]
    ty = cute.arch.thread_idx()[1]
    m = bx * cute.arch.block_dim()[0] + tx
    n = by * cute.arch.block_dim()[1] + ty
    out[m, n] = x[m, n] + x.element_type(1.0)


@cute.jit
def launch_add_one_2d_blocked(x: cute.Tensor, out: cute.Tensor, BX: int = 16, BY: int = 16):
    M = x.shape[0]
    N = x.shape[1]
    _add_one_2d_blocked_kern(x, out).launch(
        grid=((M + BX - 1) // BX, (N + BY - 1) // BY, 1),
        block=(BX, BY, 1),
    )


# ── 3×3 conv (pad=1) + ReLU + 2×2 max pool ──────────────────────────────────
# ResNet-style fused stem: x (N,C_in,H,W), weight (C_out,C_in,3,3), bias (C_out,)
# → out (N,C_out,H//2,W//2)

@cute.kernel
def _conv3x3_relu_pool_kern(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
):
    pid = cute.arch.block_idx()[0]
    H_out = out.shape[2]
    W_out = out.shape[3]
    C_in = x.shape[1]
    C_out = weight.shape[0]
    H = x.shape[2]
    W = x.shape[3]

    n = pid // (H_out * W_out)
    rest = pid % (H_out * W_out)
    h_out = rest // W_out
    w_out = rest % W_out

    for c_out in range(C_out):
        max_val = x.element_type(0.0)  # relu >= 0, 0 is safe floor
        for dh in range(2):
            for dw in range(2):
                h = h_out * 2 + dh
                w = w_out * 2 + dw
                acc = x.element_type(0.0)
                for c_in in range(C_in):
                    for kh in range(3):
                        for kw in range(3):
                            ih = h + kh - 1
                            iw = w + kw - 1
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                acc = acc + x[n, c_in, ih, iw] * weight[c_out, c_in, kh, kw]
                acc = acc + bias[c_out]
                relu_val = cute.arch.fmax(acc, x.element_type(0.0))
                max_val = cute.arch.fmax(relu_val, max_val)
        out[n, c_out, h_out, w_out] = max_val


@cute.jit
def launch_conv3x3_relu_pool(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
):
    """3×3 conv (pad=1) + bias + ReLU + 2×2 max pool (stride 2).

    x: (N,C_in,H,W), weight: (C_out,C_in,3,3), bias: (C_out,) → out: (N,C_out,H//2,W//2)
    """
    N = x.shape[0]
    H_out = x.shape[2] // 2
    W_out = x.shape[3] // 2
    _conv3x3_relu_pool_kern(x, weight, bias, out).launch(
        grid=(N * H_out * W_out, 1, 1), block=(1, 1, 1)
    )
