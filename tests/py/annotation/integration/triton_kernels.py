"""
Real Triton kernels and launch functions for E2E testing.

These are real GPU kernels that perform actual computation.
The launch functions are what tta.triton() accepts.
"""

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------------
# Kernel 1: add_one (single input → output = x + 1.0)
# ---------------------------------------------------------------------------
@triton.jit
def _add_one_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + 1.0, mask=mask)


def launch_add_one(x, out, BLOCK_SIZE=128):
    """Launch function: out = x + 1.0"""
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _add_one_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel 2: 2D elementwise add (two inputs (M,N) → output = x + y)
# ---------------------------------------------------------------------------
@triton.jit
def _add_2d_kernel(
    x_ptr, y_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_offs = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_offs = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    out_offs = offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_offs, mask=mask, other=0.0)
    tl.store(out_ptr + out_offs, x + y, mask=mask)


def launch_add_2d(x, y, out, BLOCK_M=32, BLOCK_N=32):
    """Launch function: out = x + y for 2D (M, N) tensors."""
    M, N = x.shape[0], x.shape[1]
    stride_xm, stride_xn = x.stride(0), x.stride(1)
    stride_ym, stride_yn = y.stride(0), y.stride(1)
    stride_outm, stride_outn = out.stride(0), out.stride(1)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _add_2d_kernel[grid](
        x, y, out,
        M, N,
        stride_xm=stride_xm,
        stride_xn=stride_xn,
        stride_ym=stride_ym,
        stride_yn=stride_yn,
        stride_outm=stride_outm,
        stride_outn=stride_outn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


# ---------------------------------------------------------------------------
# Kernel 3: scale (single input → output = x * 2.0)
# ---------------------------------------------------------------------------
@triton.jit
def _scale_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


def launch_scale(x, out, BLOCK_SIZE=128):
    """Launch function: out = x * 2.0"""
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _scale_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel 4: fused add + relu (two inputs → output = relu(x + y))
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_relu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    z = tl.where(z > 0, z, 0.0)
    tl.store(out_ptr + offsets, z, mask=mask)


def launch_fused_add_relu(x, y, out, BLOCK_SIZE=128):
    """Launch function: out = relu(x + y)"""
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _fused_add_relu_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel 5: GELU tanh approximation (single input → output)
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(out_ptr + offsets, y, mask=mask)


def launch_gelu(x, out, BLOCK_SIZE=128):
    """Launch function: out = gelu_tanh(x)"""
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel 6: fused multiply-add (three inputs → output = a * b + c)
# ---------------------------------------------------------------------------
@triton.jit
def _fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a * b + c, mask=mask)


def launch_fma(a, b, c, out, BLOCK_SIZE=128):
    """Launch function: out = a * b + c"""
    n = a.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _fma_kernel[grid](a, b, c, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel 7: matmul (a @ b -> out), a (M, K), b (K, N), out (M, N)
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_offs = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_offs = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    out_offs = offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def launch_matmul(a, b, out, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32):
    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]
    stride_am = K
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_outm = N
    stride_outn = 1
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a, b, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_outm, stride_outn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


# ---------------------------------------------------------------------------
# Kernel 8: 1x1 conv + bias + relu; x (N,C_in,H,W), weight (C_out,C_in), bias (C_out,), out (N,C_out,H,W)
# ---------------------------------------------------------------------------
@triton.jit
def _conv1x1_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    num_rows, H, W, C_in, C_out,
    stride_w_cout, stride_w_cin,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_rows:
        return
    n = pid // (H * W)
    rest = pid % (H * W)
    base_x = n * C_in * H * W + rest
    base_out = n * C_out * H * W + rest
    offs_cout = tl.arange(0, BLOCK_C)
    for c_out_start in range(0, C_out, BLOCK_C):
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for c_in in range(0, C_in):
            x_off = base_x + c_in * (H * W)
            x_val = tl.load(x_ptr + x_off)
            w_offs = (c_out_start + offs_cout) * stride_w_cout + c_in * stride_w_cin
            w_mask = (c_out_start + offs_cout) < C_out
            w = tl.load(weight_ptr + w_offs, mask=w_mask, other=0.0)
            acc += x_val * w
        b_offs = c_out_start + offs_cout
        b_mask = b_offs < C_out
        b = tl.load(bias_ptr + b_offs, mask=b_mask, other=0.0)
        acc += b
        acc = tl.where(acc > 0, acc, 0.0)
        out_off = base_out + (c_out_start + offs_cout) * (H * W)
        out_mask = (c_out_start + offs_cout) < C_out
        tl.store(out_ptr + out_off, acc, mask=out_mask)


def launch_conv1x1_relu(x, weight, bias, out, BLOCK_C=32):
    N, C_in, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    C_out = weight.shape[0]
    num_rows = N * H * W
    stride_w_cout = C_in
    stride_w_cin = 1
    grid = (num_rows,)
    _conv1x1_relu_kernel[grid](
        x, weight, bias, out,
        num_rows, H, W, C_in, C_out,
        stride_w_cout, stride_w_cin,
        BLOCK_C=BLOCK_C,
    )


# ---------------------------------------------------------------------------
# Kernel 9: add_one_int (int32 input → output = x + 1)
# ---------------------------------------------------------------------------
@triton.jit
def _add_one_int_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + 1, mask=mask)


def launch_add_one_int(x, out, BLOCK_SIZE=128):
    """Launch function: out = x + 1 (integer arithmetic)"""
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _add_one_int_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel: 3×3 conv (pad=1) + bias + ReLU + 2×2 max pool (stride 2)
# ResNet-style fused stem: x (N,C_in,H,W), weight (C_out,C_in,3,3), bias (C_out,)
# → out (N,C_out,H//2,W//2)
# ---------------------------------------------------------------------------
@triton.jit
def _conv3x3_relu_pool_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, H_out, W_out, H, W, C_in, C_out,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N * H_out * W_out:
        return
    n = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)
    h_out_val = hw // W_out
    w_out_val = hw % W_out

    offs_c = tl.arange(0, BLOCK_C)
    for c_start in range(0, C_out, BLOCK_C):
        c_mask = (c_start + offs_c) < C_out
        bias = tl.load(bias_ptr + c_start + offs_c, mask=c_mask, other=0.0)
        max_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)  # relu >= 0, 0 is safe floor

        for dh in range(2):
            for dw in range(2):
                h = h_out_val * 2 + dh
                w = w_out_val * 2 + dw
                acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
                for c_in in range(C_in):
                    for kh in range(3):
                        for kw in range(3):
                            ih = h + kh - 1
                            iw = w + kw - 1
                            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                            safe_ih = tl.where(valid, ih, 0)
                            safe_iw = tl.where(valid, iw, 0)
                            x_off = n * C_in * H * W + c_in * H * W + safe_ih * W + safe_iw
                            x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                            # weight layout: (C_out, C_in, 3, 3) row-major
                            w_offs = (c_start + offs_c) * (C_in * 9) + c_in * 9 + kh * 3 + kw
                            w_val = tl.load(weight_ptr + w_offs, mask=c_mask, other=0.0)
                            acc = acc + x_val * w_val
                acc = acc + bias
                relu_val = tl.where(acc > 0.0, acc, 0.0)
                max_vals = tl.where(relu_val > max_vals, relu_val, max_vals)

        out_base = n * C_out * H_out * W_out + h_out_val * W_out + w_out_val
        out_offs = out_base + (c_start + offs_c) * (H_out * W_out)
        tl.store(out_ptr + out_offs, max_vals, mask=c_mask)


def launch_conv3x3_relu_pool(x, weight, bias, out, BLOCK_C=4):
    """3×3 conv (pad=1) + bias + ReLU + 2×2 max pool (stride 2).

    x: (N,C_in,H,W), weight: (C_out,C_in,3,3), bias: (C_out,) → out: (N,C_out,H//2,W//2)
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    H_out, W_out = H // 2, W // 2
    grid = (N * H_out * W_out,)
    _conv3x3_relu_pool_kernel[grid](
        x, weight, bias, out,
        N, H_out, W_out, H, W, C_in, C_out,
        BLOCK_C=BLOCK_C,
    )


# ---------------------------------------------------------------------------
# Kernel 10: add_one_and_scale — two outputs: (x+1, x*2) from one input
# Signature: (x, out0, out1, n_elements, BLOCK_SIZE)
# ---------------------------------------------------------------------------
@triton.jit
def _add_one_and_scale_kernel(
    x_ptr, out0_ptr, out1_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out0_ptr + offsets, x + 1.0, mask=mask)
    tl.store(out1_ptr + offsets, x * 2.0, mask=mask)


def launch_add_one_and_scale(x, out0, out1, BLOCK_SIZE=128):
    """Launch function: out0 = x + 1.0,  out1 = x * 2.0  (two outputs)"""
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _add_one_and_scale_kernel[grid](x, out0, out1, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel: split_add_scale (1 input → 2 outputs: out1 = x + 1.0, out2 = x * 2.0)
#
# Used to test multi-output lower_as regions.  The region produces two escaping
# values; this kernel computes both in a single pass so the impl is IO-compatible
# (1 input, 2 outputs) and the lower_as pass can insert operator.getitem nodes.
# ---------------------------------------------------------------------------
@triton.jit
def _split_add_scale_kernel(x_ptr, out1_ptr, out2_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out1_ptr + offsets, x + 1.0, mask=mask)
    tl.store(out2_ptr + offsets, x * 2.0, mask=mask)


def launch_split_add_scale(x, out1, out2, BLOCK_SIZE=128):
    """Launch function: out1 = x + 1.0, out2 = x * 2.0 (2-output kernel)."""
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _split_add_scale_kernel[grid](x, out1, out2, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Kernel: elementwise_mul (two inputs → output = x * w)
# ---------------------------------------------------------------------------
@triton.jit
def _elementwise_mul_kernel(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) * tl.load(w_ptr + offs, mask=mask),
        mask=mask,
    )


def launch_elementwise_mul(x, w, out, BLOCK=128):
    """Launch function: out = x * w"""
    _elementwise_mul_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, w, out, x.numel(), BLOCK=BLOCK)


# ---------------------------------------------------------------------------
# Real-world kernels: used by test_real_world_e2e.py
# ---------------------------------------------------------------------------

# SwiGLU: out = gate * silu(gate) — LLaMA/Mistral FFN, fused in one kernel.
# TRT would emit silu(gate) and elementwise-mul as two separate ops;
# this kernel reads gate once and writes out once (better L2 reuse).
@triton.jit
def _swiglu_kernel(gate_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask)
    silu = gate / (1.0 + tl.exp(-gate))  # gate * sigmoid(gate)
    tl.store(out_ptr + offs, gate * silu, mask=mask)


def launch_swiglu(gate, out, BLOCK=256):
    """Launch function: out = gate * silu(gate)  (fused SwiGLU gate)"""
    _swiglu_kernel[(triton.cdiv(gate.numel(), BLOCK),)](gate, out, gate.numel(), BLOCK=BLOCK)


# RMS Norm: out = x / rms(x) * w — LLaMA/Mistral normalization.
# Single-block kernel: processes the full vector in one CTA (BLOCK >= n).
@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    mean_sq = tl.sum(x * x) / n
    rms = tl.sqrt(mean_sq + 1e-6)
    tl.store(out_ptr + offs, x / rms * w, mask=mask)


def launch_rms_norm(x, w, out, BLOCK=256):
    """Launch function: out = x / rms(x) * w  (1 CTA, BLOCK >= n)"""
    _rms_norm_kernel[(1,)](x, w, out, x.numel(), BLOCK=BLOCK)


# GLU (Gated Linear Unit): out = sigmoid(gate) * value — SwiGLU FFN gate.
@triton.jit
def _glu_kernel(gate_ptr, value_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs,
        tl.sigmoid(tl.load(gate_ptr + offs, mask=mask)) * tl.load(value_ptr + offs, mask=mask),
        mask=mask)


def launch_glu(gate, value, out, BLOCK=256):
    """Launch function: out = sigmoid(gate) * value"""
    _glu_kernel[(triton.cdiv(gate.numel(), BLOCK),)](gate, value, out, gate.numel(), BLOCK=BLOCK)


# GEGLU (GeLU-Gated Linear Unit): out = fast_gelu(gate) * value — PaLM/T5v1.1 FFN.
# Fast GeLU: gate * sigmoid(1.702 * gate)
@triton.jit
def _geglu_kernel(gate_ptr, value_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask)
    value = tl.load(value_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, gate * tl.sigmoid(1.702 * gate) * value, mask=mask)


def launch_geglu(gate, value, out, BLOCK=256):
    """Launch function: out = fast_gelu(gate) * value  (GEGLU)"""
    _geglu_kernel[(triton.cdiv(gate.numel(), BLOCK),)](gate, value, out, gate.numel(), BLOCK=BLOCK)


# Per-element affine: out = x * scale + offset — custom dequantization.
@triton.jit
def _affine_kernel(x_ptr, scale_ptr, offset_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.load(scale_ptr + offs, mask=mask)
    o = tl.load(offset_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * s + o, mask=mask)


def launch_affine(x, scale, offset, out, BLOCK=256):
    """Launch function: out = x * scale + offset"""
    _affine_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, scale, offset, out, x.numel(), BLOCK=BLOCK)


# Bias add: out = x + bias — ALiBi / learned positional bias.
@triton.jit
def _bias_add_kernel(x_ptr, bias_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) + tl.load(bias_ptr + offs, mask=mask),
        mask=mask)


def launch_bias_add(x, bias, out, BLOCK=256):
    """Launch function: out = x + bias"""
    _bias_add_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, bias, out, x.numel(), BLOCK=BLOCK)


# Expert broadcast bias: out[i,j] = x[i,j] + bias[j] — used in MoE expert add.
# x and out are passed as flat N*D pointers; bias is broadcast across the N tokens.
@triton.jit
def _expert_add_kernel(x_ptr, bias_ptr, out_ptr, n, d, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + (offs % d), mask=mask)
    tl.store(out_ptr + offs, x + b, mask=mask)


def launch_expert_add(x, bias, out, BLOCK=256):
    """Launch function: out[i,j] = x[i,j] + bias[j]  (broadcast bias over N tokens)"""
    n_total = x.numel()
    d = bias.shape[0]
    _expert_add_kernel[(triton.cdiv(n_total, BLOCK),)](x, bias, out, n_total, d, BLOCK=BLOCK)


# Fused MoE top-1 bias expert: for each token, compute gate logits, pick the
# top-1 expert, and add that expert's bias.  One CTA per token; BLOCK_D must
# cover the hidden dimension.  Replaces the sparse dispatch loop (nonzero +
# torch.where) with a fully fused, shape-static kernel.
@triton.jit
def _moe_bias_kernel(x_ptr, gw_ptr, b0_ptr, b1_ptr, out_ptr,
                     N, D, BLOCK_D: tl.constexpr):
    pid  = tl.program_id(0)          # one CTA per token
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x   = tl.load(x_ptr  + pid * D + offs, mask=mask)
    gw0 = tl.load(gw_ptr + 0   * D + offs, mask=mask)
    gw1 = tl.load(gw_ptr + 1   * D + offs, mask=mask)

    # Gate logits: dot(x, gate_weight[k])
    g0 = tl.sum(x * gw0, axis=0)
    g1 = tl.sum(x * gw1, axis=0)

    # Top-1 routing: pick argmax (no softmax needed for argmax)
    use_1 = g1 > g0

    b0 = tl.load(b0_ptr + offs, mask=mask)
    b1 = tl.load(b1_ptr + offs, mask=mask)
    bias = tl.where(use_1, b1, b0)

    tl.store(out_ptr + pid * D + offs, x + bias, mask=mask)


def launch_moe_bias(x, gate_weight, bias0, bias1, out, BLOCK_D=64):
    """Fused MoE top-1: out[i] = x[i] + bias_{argmax(x[i] @ gate_weight.T)}"""
    N, D = x.shape
    _moe_bias_kernel[(N,)](x, gate_weight, bias0, bias1, out, N, D, BLOCK_D=BLOCK_D)


# Fused bias + ReLU: out = relu(x + bias) — post-linear activation.
@triton.jit
def _bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x + b, 0.0), mask=mask)


def launch_bias_relu(x, bias, out, BLOCK=256):
    """Launch function: out = relu(x + bias)"""
    _bias_relu_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, bias, out, x.numel(), BLOCK=BLOCK)


# Hard Swish: out = x * clamp(x+3, 0, 6) / 6 — MobileNetV3 / EfficientNet.
@triton.jit
def _hard_swish_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    relu6 = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    tl.store(out_ptr + offs, x * relu6 / 6.0, mask=mask)


def launch_hard_swish(x, out, BLOCK=256):
    """Launch function: out = x * clamp(x+3, 0, 6) / 6  (hard swish)"""
    _hard_swish_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, out, x.numel(), BLOCK=BLOCK)


# Temperature-scaled exp: out = exp(x / temperature) — attention score scaling.
# temperature is a 0-d scalar nn.Parameter (single float).
@triton.jit
def _temp_exp_kernel(x_ptr, temp_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    t = tl.load(temp_ptr + 0)
    tl.store(out_ptr + offs, tl.exp(x / t), mask=mask)


def launch_temp_exp(x, temp, out, BLOCK=256):
    """Launch function: out = exp(x / temperature)"""
    _temp_exp_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, temp, out, x.numel(), BLOCK=BLOCK)


# Embedding scale: out = x * scale — Transformer sqrt(d_model) embedding scale.
# scale is a 0-d scalar nn.Parameter (single float).
@triton.jit
def _embed_scale_kernel(x_ptr, scale_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.load(scale_ptr + 0)
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * s, mask=mask)


def launch_embed_scale(x, scale, out, BLOCK=256):
    """Launch function: out = x * scale (scalar parameter)"""
    _embed_scale_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, scale, out, x.numel(), BLOCK=BLOCK)


if __name__ == "__main__":
    # Quick sanity check
    x = torch.randn(256, device="cuda", dtype=torch.float32)

    out = torch.empty_like(x)
    launch_add_one(x, out)
    diff = (out - (x + 1.0)).abs().max().item()
    print(f"add_one max diff: {diff}, PASS: {diff < 1e-6}")

    out = torch.empty_like(x)
    launch_scale(x, out)
    diff = (out - (x * 2.0)).abs().max().item()
    print(f"scale max diff: {diff}, PASS: {diff < 1e-6}")
