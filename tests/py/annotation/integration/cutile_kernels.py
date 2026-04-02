"""
CuTile kernels and launch functions for E2E testing.

Requires cuda.tile (pip install cuda-tile).
"""

from __future__ import annotations

import torch

import cuda.tile as ct


@ct.kernel
def add_one_tile(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    result = x_tile + 1.0
    ct.store(out, index=(pid,), tile=result)


def launch_add_one(x, out, BLOCK=128):
    n = x.numel()
    grid = (ct.cdiv(n, BLOCK), 1, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(stream, grid, add_one_tile, (x, out, BLOCK))


@ct.kernel
def add_2d_tile(x, y, out, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    x_tile = ct.load(x, index=(pid_m, pid_n), shape=(tile_m, tile_n))
    y_tile = ct.load(y, index=(pid_m, pid_n), shape=(tile_m, tile_n))
    result = x_tile + y_tile
    ct.store(out, index=(pid_m, pid_n), tile=result)


def launch_add_2d(x, y, out, BLOCK_M=32, BLOCK_N=32):
    M, N = x.shape[0], x.shape[1]
    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(stream, grid, add_2d_tile, (x, y, out, BLOCK_M, BLOCK_N))


@ct.kernel
def fused_add_relu_tile(x, y, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    y_tile = ct.load(y, index=(pid,), shape=(tile_size,))
    z = x_tile + y_tile
    result = ct.where(z > 0.0, z, 0.0)
    ct.store(out, index=(pid,), tile=result)


def launch_fused_add_relu(x, y, out, BLOCK=128):
    n = x.numel()
    grid = (ct.cdiv(n, BLOCK), 1, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(stream, grid, fused_add_relu_tile, (x, y, out, BLOCK))


@ct.kernel
def gelu_tile(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    c = 0.7978845608028654
    inner = c * (x_tile + 0.044715 * x_tile * x_tile * x_tile)
    t = ct.tanh(inner)
    y = 0.5 * x_tile * (1.0 + t)
    ct.store(out, index=(pid,), tile=y)


def launch_gelu(x, out, BLOCK=128):
    n = x.numel()
    grid = (ct.cdiv(n, BLOCK), 1, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(stream, grid, gelu_tile, (x, out, BLOCK))


@ct.kernel
def scale_tile(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    result = x_tile * 2.0
    ct.store(out, index=(pid,), tile=result)


def launch_scale(x, out, BLOCK=128):
    n = x.numel()
    grid = (ct.cdiv(n, BLOCK), 1, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(stream, grid, scale_tile, (x, out, BLOCK))


@ct.kernel
def fma_tile(a, b, c, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    c_tile = ct.load(c, index=(pid,), shape=(tile_size,))
    result = a_tile * b_tile + c_tile
    ct.store(out, index=(pid,), tile=result)


def launch_fma(a, b, c, out, BLOCK=128):
    n = a.numel()
    grid = (ct.cdiv(n, BLOCK), 1, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(a, "is_cuda", True) else None
    ct.launch(stream, grid, fma_tile, (a, b, c, out, BLOCK))


@ct.kernel
def matmul_tile(a, b, out, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    a_val = ct.load(a, index=(pid_m, 0), shape=())
    b_val = ct.load(b, index=(0, pid_n), shape=())
    acc = a_val * b_val
    for k in range(1, K):
        a_val = ct.load(a, index=(pid_m, k), shape=())
        b_val = ct.load(b, index=(k, pid_n), shape=())
        acc = acc + a_val * b_val
    ct.store(out, index=(pid_m, pid_n), tile=acc)


def launch_matmul(a, b, out, BLOCK_M=1, BLOCK_N=1):
    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]
    grid = (M, N, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(a, "is_cuda", True) else None
    ct.launch(stream, grid, matmul_tile, (a, b, out, M, N, K))


@ct.kernel
def conv1x1_relu_tile(
    x, w, bias, out,
    num_rows: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    C_in: ct.Constant[int],
    C_out: ct.Constant[int],
):
    pid = ct.bid(0)
    if pid >= num_rows:
        return
    n = pid // (H * W)
    rest = pid % (H * W)
    base_x = n * C_in * H * W + rest
    base_out = n * C_out * H * W + rest
    for c_out in range(C_out):
        acc = 0.0
        for c_in in range(C_in):
            x_off = base_x + c_in * (H * W)
            w_off = c_out * C_in + c_in
            x_val = ct.load(x, index=(x_off,), shape=(1,))
            w_val = ct.load(w, index=(w_off,), shape=(1,))
            acc += x_val.item() * w_val.item()
        b_val = ct.load(bias, index=(c_out,), shape=(1,))
        acc += b_val.item()
        acc = max(acc, 0.0)
        out_off = base_out + c_out * (H * W)
        ct.store(out, index=(out_off,), tile=acc)


def launch_conv1x1_relu(x, weight, bias, out, BLOCK_C=32):
    N, C_in, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    C_out = weight.shape[0]
    num_rows = N * H * W
    grid = (num_rows, 1, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(stream, grid, conv1x1_relu_tile, (x, weight, bias, out, num_rows, H, W, C_in, C_out))


# ---------------------------------------------------------------------------
# Kernel: add_one_int (int32 input → output = x + 1)
# ---------------------------------------------------------------------------
@ct.kernel
def add_one_int_tile(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    result = x_tile + 1
    ct.store(out, index=(pid,), tile=result)


def launch_add_one_int(x, out, BLOCK=128):
    """Launch function: out = x + 1 (integer arithmetic)"""
    n = x.numel()
    grid = (ct.cdiv(n, BLOCK), 1, 1)
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(stream, grid, add_one_int_tile, (x, out, BLOCK))


# ---------------------------------------------------------------------------
# Kernel: 3×3 conv (pad=1) + bias + ReLU + 2×2 max pool (stride 2)
# ResNet-style fused stem: x (N,C_in,H,W), weight (C_out,C_in,3,3), bias (C_out,)
# → out (N,C_out,H//2,W//2)
# ---------------------------------------------------------------------------
@ct.kernel
def conv3x3_relu_pool_tile(
    x, weight, bias, out,
    N: ct.Constant[int],
    H_out: ct.Constant[int],
    W_out: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    C_in: ct.Constant[int],
    C_out: ct.Constant[int],
):
    pid = ct.bid(0)
    if pid >= N * H_out * W_out:
        return
    n = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    for c_out_idx in range(C_out):
        max_val = 0.0  # relu >= 0, 0 is safe floor
        for dh in range(2):
            for dw in range(2):
                h = h_out * 2 + dh
                w_in = w_out * 2 + dw
                acc = 0.0
                for c_in in range(C_in):
                    for kh in range(3):
                        for kw in range(3):
                            ih = h + kh - 1
                            iw = w_in + kw - 1
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                x_off = n * C_in * H * W + c_in * H * W + ih * W + iw
                                # weight layout: (C_out, C_in, 3, 3) row-major
                                wt_off = c_out_idx * (C_in * 9) + c_in * 9 + kh * 3 + kw
                                x_val = ct.load(x, index=(x_off,), shape=(1,))
                                wt_val = ct.load(weight, index=(wt_off,), shape=(1,))
                                acc += x_val.item() * wt_val.item()
                b_val = ct.load(bias, index=(c_out_idx,), shape=(1,))
                acc += b_val.item()
                relu_val = max(acc, 0.0)
                if relu_val > max_val:
                    max_val = relu_val
        out_off = n * C_out * H_out * W_out + c_out_idx * H_out * W_out + h_out * W_out + w_out
        ct.store(out, index=(out_off,), tile=max_val)


def launch_conv3x3_relu_pool(x, weight, bias, out, BLOCK=1):
    """3×3 conv (pad=1) + bias + ReLU + 2×2 max pool (stride 2).

    x: (N,C_in,H,W), weight: (C_out,C_in,3,3), bias: (C_out,) → out: (N,C_out,H//2,W//2)
    """
    N, C_in, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    C_out = weight.shape[0]
    H_out, W_out = H // 2, W // 2
    stream = torch.cuda.current_stream().cuda_stream if getattr(x, "is_cuda", True) else None
    ct.launch(
        stream, (N * H_out * W_out, 1, 1), conv3x3_relu_pool_tile,
        (x, weight, bias, out, N, H_out, W_out, H, W, C_in, C_out),
    )
