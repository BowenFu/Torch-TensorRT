# Torch-TensorRT Annotation Layer (TTA)

Use TTA when you want **per-region** control (precision, quantization, validation, etc.): Torch-TRT has no per-region config — it is global only. TTA also helps when you need to force a macro op or custom plugin, annotate without editing source, trace export-hostile code, or profile multiple impls.

```python
import torch_tensorrt.annotation as tta
```

---

## Table of contents

**Start small**
1. [When Torch-TRT would not fuse: force one op (lower_as)](#1-when-torch-trt-would-not-fuse-force-one-op-lower_as)
2. [When Torch-TRT has no per-region precision: autocast](#2-when-torch-trt-has-no-per-region-precision-autocast)
3. [When Torch-TRT has no per-region control: multiple regions](#3-when-torch-trt-has-no-per-region-control-multiple-regions)

**Real world**
4. [Eager as reference](#4-eager-as-reference)
5. [Compile: dynamic shapes and precisions](#5-compile-dynamic-shapes-and-precisions)
6. [Annotate a model you don't own](#6-annotate-a-model-you-dont-own)
7. [Code that can't be traced: export_as](#7-code-that-cant-be-traced-export_as)
8. [Quantize a region](#8-quantize-a-region)
9. [Validate regions in CI](#9-validate-regions-in-ci)
10. [Autotune: pick best implementation](#10-autotune-pick-best-implementation)

**Reference**
11. [Implementation types](#11-implementation-types)
12. [Quick reference](#12-quick-reference)
13. [Troubleshooting & limitations](#13-troubleshooting--limitations)
14. [Running tests](#14-running-tests)

---

## Start small

### 1. When Torch-TRT would not fuse: force one op (lower_as)

**Requirements:** PyTorch (with `torch.export`), TensorRT, torch-tensorrt, CUDA. Optional: Triton (for custom plugin).

**Why TTA:** Torch-TRT alone would lower `relu(x + y)` as **two** TRT ops (elementwise add + activation). You want **one** fused op (e.g. a Triton kernel or TRT macro) for better perf. **lower_as** forces this region to the impl you provide; without it, you get the default graph.

**Example: fused add+ReLU.** Region = exactly `relu(x + y)`; impl = custom Triton plugin. Torch-TRT alone cannot produce that single plugin — it would emit add + relu.

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta
# From tests/py/annotation/integration/triton_kernels import launch_fused_add_relu

class ResidualAddReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        with tta.lower_as(
            impl=tta.custom_plugin(
                tta.triton(launch_fused_add_relu, configs=[{"BLOCK_SIZE": 256}])
            ),
            require=True,
            name="fused_add_relu",
        ):
            x = torch.relu(x + identity)
        return x

# model = ResidualAddReLU().cuda().eval()
# x = torch.randn(2, 64, 8, 8, device="cuda")
# trt_model = tta.compile(model, inputs=(x,))
# out = trt_model(x)
```

Use **launch_fused_add_relu** from `tests/py/annotation/integration/triton_kernels.py` (or your own Triton launch). The region must match the impl: here, one input is the add result, so the region is exactly `relu(x + identity)`.

**Real-world Example:** You want the attention block to become the TRT **add_attention** macro op. Torch-TRT alone would lower attention as **many** primitives (matmuls, softmax, scale). With **lower_as(tta.builtin("add_attention", ...))** you force one macro op. The region must be your full attention reference (Q/K/V, mask, etc.); the impl must match that IO. Requires a TensorRT build that exposes `add_attention`.

---

### 2. When Torch-TRT has no per-region precision: autocast

**Why TTA:** When you want per-region precision (e.g. backbone in fp16, stem/head in fp32), Torch-TRT alone cannot help — it only supports **global** precision. **tta.autocast** scopes precision per region.

```python
def forward(self, x):
    x = self.stem(x)
    with tta.autocast(mode="fp16"):
        x = self.backbone(x)
    return self.head(x)
```

**Real-world Example:** Run transformer layers in fp16 for speed; keep embedding and LM head in fp32 for numerical stability and logit quality. Runnable end-to-end:

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta

class TinyLLM(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids).float()
        with tta.autocast(mode="fp16"):
            for layer in self.layers:
                x = x + torch.relu(layer(x))
        x = self.norm(x.float())
        logits = self.lm_head(x)
        return logits

model = TinyLLM().cuda().eval()
input_ids = torch.randint(0, 32000, (2, 8), device="cuda")
trt_model = tta.compile(model, inputs=(input_ids,), enabled_precisions={torch.float16})
out = trt_model(input_ids)
```

---

### 3. When Torch-TRT has no per-region control: multiple regions

**Why TTA:** When you want different precision per block (this block fp16, that block fp32), Torch-TRT has no per-region config. **tta.autocast** applies precision per region.

**Example:** One region in fp16 — Torch-TRT alone would make the whole graph fp16 or fp32.

```python
def forward(self, x):
    x = self.stem(x)
    with tta.autocast(mode="fp16"):
        x = self.backbone(x)
    return self.head(x)
```


**Real-world Example:** You want transformer layers in fp16 and embedding/LM head in fp32; Torch-TRT has no per-region precision. Runnable end-to-end:

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta

class TinyLLM(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids).float()
        with tta.autocast(mode="fp16"):
            for layer in self.layers:
                x = x + torch.relu(layer(x))
        x = self.norm(x.float())
        logits = self.lm_head(x)
        return logits

model = TinyLLM().cuda().eval()
input_ids = torch.randint(0, 32000, (2, 8), device="cuda")
trt_model = tta.compile(model, inputs=(input_ids,), enabled_precisions={torch.float16})
out = trt_model(input_ids)
```

---

## Real world

### 4. Eager as reference

**Why TTA:** Torch-TRT does not define a single eager reference path for “this block will become that TRT op.” TTA guarantees annotations are no-ops in eager: the Python body is the full reference, so accuracy validation (eager vs TRT) is well-defined and you can diff region behavior.

**Three constraints:**

1. **Eager is the full reference.** All TTA surfaces (including `tta.lower_as` + `tta.builtin`) must not change Python semantics. Removing annotations must not change eager outputs.
2. **No hidden capture.** Every constant that TensorRT needs must be passed explicitly via `tta.builtin(...)` kwargs. The binder uses those when calling the TRT add_* API; region boundary IO is only true runtime tensors.
3. **`tta.lower_as` only affects import.** It tags region→impl metadata; `torch.export` still sees the full primitive graph for the region (good for debugging and accuracy diffs).

**Pattern:** Keep module attributes/buffers as usual and use them in the eager body. Pass the **same** objects as kwargs in `tta.builtin(...)`. Eager runs your Python; at compile, the spec kwargs feed the TRT layer.

**RoPE example (runnable):** Eager runs a Python RoPE reference; the same tensors are passed into `tta.builtin` so there is no hidden capture. Requires a TensorRT build that exposes `add_rotary_embedding`; otherwise the `tta.builtin(...)` call in `__init__` will raise. Use **require=False** so compile can fall back to the Python ref if the macro is not available.

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta

def apply_rope_ref(q, cos_cache, sin_cache, position_ids):
    batch, seq, head_dim = q.shape
    cos = cos_cache[position_ids].squeeze(2)
    sin = sin_cache[position_ids].squeeze(2)
    q0, q1 = q.chunk(2, dim=-1)
    return torch.stack([q0 * cos - q1 * sin, q1 * cos + q0 * sin], dim=-1).flatten(-2)

class RoPEBlock(nn.Module):
    def __init__(self, head_dim, max_seq=128):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq, device=inv_freq.device).float()
        freqs = torch.outer(t, inv_freq)
        cos_cache = freqs.cos().unsqueeze(0).unsqueeze(-1)
        sin_cache = freqs.sin().unsqueeze(0).unsqueeze(-1)
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)
        self.rope_impl = tta.builtin(
            "add_rotary_embedding",
            cos_cache=self.cos_cache,
            sin_cache=self.sin_cache,
            interleaved=False,
            rotary_embedding_dim=self.head_dim,
        )

    def forward(self, q, position_ids):
        with tta.lower_as(impl=self.rope_impl, require=False, name="rope_block"):
            q_rot = apply_rope_ref(q, self.cos_cache, self.sin_cache, position_ids)
        return q_rot

model = RoPEBlock(head_dim=64).cuda().eval()
q = torch.randn(2, 8, 64, device="cuda")
position_ids = torch.arange(8, device="cuda").unsqueeze(0).expand(2, 8)
out = model(q, position_ids)
trt_model = tta.compile(model, inputs=(q, position_ids))
out_trt = trt_model(q, position_ids)
```

- **Eager:** `forward` runs `apply_rope_ref(...)` with real `self.cos_cache` / `self.sin_cache` / `position_ids`. Same outputs with or without the annotation.
- **No hidden capture:** Constants for `add_rotary_embedding` come from `BuiltinSpec.kwargs`; boundary IO is `q`, `position_ids`.
- **Export:** Region is still full RoPE math in the FX graph; `tta.lower_as` only tags it for lowering.

**Same pattern for other macro-ops:**

- **add_attention** — Eager: full SDPA in PyTorch. Builtin: `tta.builtin("add_attention", norm_op=..., causal=...)` (scalars only). Region IO: `query`, `key`, `value`, optional mask.
- **add_kv_cache_update** — Eager: explicit scatter into cache. Builtin: `tta.builtin("add_kv_cache_update", cache_mode=...)`. Region IO: `cache`, `update`, `write_indices`.
- **add_moe** — Eager: full top-k routing + expert MLPs + combine. Builtin: `tta.builtin("add_moe", ...)` with static config; expert weights etc. in the module and used in the ref body. Region IO: `hidden_states`, `selected_experts_for_tokens`, `scores_for_selected_experts`.

---

### 5. Compile: dynamic shapes and precisions

**Why TTA:** `tta.compile` runs TTA’s export and region lowering, then invokes torch-trt; you need it whenever you use annotations. For shapes and precisions it forwards the same options (example tensors, `Input` min/opt/max, `enabled_precisions`) to the underlying compiler.

**Dynamic (min/opt/max):**

```python
trt_model = tta.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=(1, 3, 224, 224),
        opt_shape=(4, 3, 224, 224),
        max_shape=(8, 3, 224, 224),
    )],
)
```

**Real-world Example:** Decoder with dynamic batch and sequence length; set min/opt/max for `[batch, seq, hidden]`. Runnable end-to-end:

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta
import torch_tensorrt

class LLMDecoder(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(2)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = hidden_states + torch.relu(layer(hidden_states))
        return hidden_states

model = LLMDecoder().cuda().eval()
trt_model = tta.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=(1, 1, 4096),
            opt_shape=(4, 512, 4096),
            max_shape=(8, 2048, 4096),
        ),
    ],
    enabled_precisions={torch.float16},
)
x = torch.randn(2, 64, 4096, device="cuda", dtype=torch.float16)
out = trt_model(x)
```

**Precisions:** `tta.compile(model, inputs=(x,), enabled_precisions={torch.float16})`.

---

### 6. Annotate a model you don't own

**Why TTA:** When you want per-region config (e.g. fp16 or lower_as only on `model.encoder`) on a model you don't own, Torch-TRT has no way to scope by region without editing source. **tta.annotate(model)** attaches annotations by submodule instance or predicate, patching forward methods in-place; then compile the original model.

**Real-world Example:** Run all transformer layers in fp16 without editing the model class; annotate by submodule. Runnable end-to-end with a small LLM-like model:

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta

class TinyLLM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(128, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, 32)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + torch.relu(layer(x))
        return self.lm_head(x)

model = TinyLLM().cuda().eval()
ann = tta.annotate(model)
ann.autocast(where=list(model.layers), mode="fp16")
x = torch.randn(2, 4, 128, device="cuda")
trt_model = tta.compile(model, inputs=(x,), enabled_precisions={torch.float16})
out = trt_model(x)
```

**where:** `model.encoder`, `list(model.layers)`, or predicate `lambda m: isinstance(m, nn.Linear)`. **annotate_on:** `AnnotateOn.MODULE_INSTANCE` (default) or `AnnotateOn.MODULE_PATH`.

---

### 7. Code that can't be traced: export_as

**Why TTA:** Torch-TRT (and torch.export) cannot trace code that uses `.item()` in a branch or unsupported ops like `nn.GRU`; export fails. **tta.export_as** replaces the callable with a single leaf so the graph is traceable and the call is implemented by your TRT layer (builtin or plugin).

**Real-world Example:** Branch on a tensor value (e.g. seq length or a flag) forces Python control flow; export cannot trace it. Wrap that logic in a callable and replace it with a TRT impl. Runnable end-to-end:

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta

class ModelWithBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    @staticmethod
    @tta.export_as(impl=tta.builtin("add_scale", scale=2.0, shift=0.0))
    def maybe_scale(x):
        if x.sum().item() > 0.0:
            return x * 2.0
        return x

    def forward(self, x):
        return self.maybe_scale(self.fc(x))

model = ModelWithBranch().cuda().eval()
x = torch.randn(2, 64, device="cuda")
trt_model = tta.compile(model, inputs=(x,))
out = trt_model(x)
```

**Unsupported op (e.g. nn.GRU):** Export doesn’t support GRU/LSTM. Wrap the step in a callable and replace with a TRT plugin (you must register the plugin). Complete class; replace the plugin name with your registered one:

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta

class ModelWithGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(64, 64)
        self.gru = nn.GRU(64, 128, batch_first=True)

    @tta.export_as(impl=tta.plugin("MyGRUPlugin", "1.0", ""))
    def gru_step(self, x):
        return self.gru(x)[0]

    def forward(self, x):
        return self.gru_step(self.encoder(x))

model = ModelWithGRU().cuda().eval()
x = torch.randn(2, 10, 64, device="cuda")
# trt_model = tta.compile(model, inputs=(x,))  # requires MyGRUPlugin registered
```

If output shape isn’t obvious, provide **meta_impl**.

---

### 8. Quantize a region

**Why TTA:** When you want per-region quantization (e.g. int8 only for specific layers), Torch-TRT has no per-region config — it applies quantization globally or by pattern. **tta.quantize** scopes quantization to the wrapped region. ModelOpt runs at compile. Use **require=False**; after export, module boundaries are lost so the module-centric path may not find modules.

```python
def forward(self, x):
    for layer in self.layers:
        with tta.quantize(mode="int8", require=False):
            x = layer(x)
    return x
```

**Real-world Example:** Quantize only the FFN to int8 for throughput; keep the rest in fp16. Runnable end-to-end (quantize region may fall back if no eligible modules; use require=False):

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta

class TinyDecoderWithFFNQuant(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.ffn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        with tta.autocast(mode="fp16"):
            hidden_states = hidden_states + torch.relu(self.attn(hidden_states))
        with tta.quantize(mode="int8", require=False):
            hidden_states = hidden_states + self.ffn(hidden_states)
        return hidden_states

model = TinyDecoderWithFFNQuant().cuda().eval()
x = torch.randn(2, 8, 64, device="cuda")
trt_model = tta.compile(model, inputs=(x,), enabled_precisions={torch.float16})
out = trt_model(x)
```

---

### 9. Validate regions in CI

**Why TTA:** When you want per-region validation in CI (e.g. “this block became one layer”, “this region is fp16 only”), Torch-TRT has no per-region config. **tta.expect(name, checks)** plus **evaluate_all_regions(trt_model)** gives you pass/fail per named region (layer count, allowed/forbidden dtypes).

```python
def forward(self, x):
    with tta.expect("encoder", checks=[
        tta.check.item_count(min=1),
        tta.check.precision_allowed({"fp16", "fp32"}),
        tta.check.precision_forbidden({"int8"}),
    ]):
        return self.encoder(x)

trt_model = tta.compile(model, inputs=(x.cuda(),))
from torch_tensorrt.annotation._expect._evaluate import evaluate_all_regions
results = evaluate_all_regions(trt_model)
assert all(r.status in ("PASS", "SKIP") for r in results)
```

**Real-world Example:** In CI, assert that the first layer’s region became at least one TRT layer and is fp16. Runnable end-to-end:

```python
import torch
import torch.nn as nn
import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation._expect._evaluate import evaluate_all_regions

class TinyDecoderWithExpect(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, hidden_states):
        with tta.expect("attn_layer0", checks=[
            tta.check.item_count(min=1),
            tta.check.precision_allowed({"fp16", "fp32"}),
            tta.check.precision_forbidden({"int8"}),
        ]):
            hidden_states = hidden_states + torch.relu(self.layers[0](hidden_states))
        for layer in self.layers[1:]:
            hidden_states = hidden_states + torch.relu(layer(hidden_states))
        return hidden_states

model = TinyDecoderWithExpect().cuda().eval()
x = torch.randn(2, 8, 64, device="cuda")
trt_model = tta.compile(model, inputs=(x,), enabled_precisions={torch.float16})
results = evaluate_all_regions(trt_model)
assert all(r.status in ("PASS", "SKIP") for r in results)
out = trt_model(x)
```

---

### 10. Autotune: pick best implementation

**Why TTA:** Torch-TRT chooses one implementation per op; you cannot profile several candidates (e.g. Triton kernel vs plugin vs baseline) and rewrite the program with the winner. **tta.autotune** marks the region with multiple impls; a separate step compiles and profiles each (and optionally the baseline), then rewrites with the best. Use when you only want to replace the baseline if it’s not the fastest.

```python
with tta.autotune(
    impls=[impl_flash_attn, impl_builtin_attn],
    include_baseline_in_search=True,
    name="self_attention",
):
    # Baseline: TRT decomposes into matmul, scale, mask, softmax, matmul (many ops)
    # impl_flash_attn: single fused Triton Flash Attention kernel
    # impl_builtin_attn: TRT add_attention macro (single fused op)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Real-world Example:** Llama-style FFN with RMSNorm autotune. The RMSNorm region is a multi-op subgraph (cast to fp32, square, mean, add eps, rsqrt, multiply by weight) — TRT baseline decomposes it into 5–6 separate layers. A fused Triton RMSNorm kernel replaces all of that with a single launch. Autotune compiles both whole-engine and picks the faster one:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation._autotune import run_autotune

@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    tl.store(out_ptr + row * stride + offs, (x / tl.sqrt(var + eps) * w).to(x.dtype), mask=mask)

def launch_rms_norm(x, weight, out, BLOCK_N=1024):
    M = x.numel() // x.shape[-1]
    _rms_norm_kernel[(M,)](x, weight, out, x.stride(-2), x.shape[-1], 1e-6, BLOCK_N=BLOCK_N)

impl_fused_rmsnorm = tta.custom_plugin(
    tta.triton(launch_rms_norm, configs=[{"BLOCK_N": 1024}, {"BLOCK_N": 512}])
)

class LlamaFFN(nn.Module):
    def __init__(self, hidden=4096):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden))
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj   = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def _rms_norm(self, x):
        return x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm_weight

    def forward(self, hidden_states):
        with tta.autotune(
            impls=[impl_fused_rmsnorm],
            include_baseline_in_search=True,
            name="ffn_rmsnorm",
        ):
            normed = self._rms_norm(hidden_states)
        return self.down_proj(torch.silu(self.gate_proj(normed)) * self.up_proj(normed))

model = LlamaFFN().cuda().eval()
x = torch.randn(2, 512, 4096, device="cuda")
```

Workflow: export with TTA (so EP carries `ep._tta`), then run **run_autotune(ep, inputs)** which returns **(trt_module, decisions)**. `decisions[rid]["choice"]` is `"impl"` or `"baseline"`; `decisions[rid]["metric"]` is end-to-end latency in ms.

```python
from torch_tensorrt.annotation._autotune import run_autotune
from torch_tensorrt.annotation._capture_state import (
    CAPTURE_MODE, CaptureMode, set_capture_mode,
    install_graph_tagging, uninstall_graph_tagging,
    get_region_table, get_impl_registry, reset_region_table,
)
from torch_tensorrt.annotation._compile.pipeline import build_annotation_ir

_cm = CAPTURE_MODE.set(CaptureMode(kind="amp"))
set_capture_mode(True)
install_graph_tagging()
ep = torch.export.export(model, (x,))
uninstall_graph_tagging()
set_capture_mode(False)
CAPTURE_MODE.reset(_cm)

build_annotation_ir(ep, ep.module(), get_region_table())
ep._tta["impl_registry"] = get_impl_registry()
reset_region_table()

trt_mod, decisions = run_autotune(ep, inputs=[x])
```

---

## Reference

### 11. Implementation types

Used as `impl=...` in **tta.lower_as** or **tta.export_as**.

- **tta.builtin** — TRT `INetworkDefinition.add_*` (e.g. `add_attention`, `add_rotary_embedding`, `add_kv_cache_update`, `add_moe`, or `add_activation`, `add_scale`). Same name and kwargs as TRT Python API.
- **tta.plugin** — Registered C++ plugin: `tta.plugin("Name", "Version", "Namespace", ...)`. **tta.self_attr("module.path")** for module tensors as inputs.
- **tta.custom_plugin** — AOT kernels: **tta.triton(launch_fn, configs=[...])**, **tta.cutile(...)** (Blackwell), **tta.cutedsl(...)**.

```python
tta.builtin("add_activation", type=0)
tta.builtin("add_scale", scale=2.0, shift=tta.self_attr("bias"))
tta.plugin("InstanceNormalization_TRT", "3", "", epsilon=1e-5, scales=tta.self_attr("norm.weight"), bias=tta.self_attr("norm.bias"))
tta.custom_plugin(tta.triton(launch_kernel, configs=[{"BLOCK": 128}, {"BLOCK": 256}]))
```

---

### 12. Quick reference

| Goal | Use | Snippet |
|------|-----|--------|
| Block → one TRT op | **tta.lower_as** | `with tta.lower_as(impl=..., require=True, name="..."):` |
| Code can't be traced | **tta.export_as** | `@tta.export_as(impl=...)` on callable |
| fp16/bf16 region | **tta.autocast** | `with tta.autocast(mode="fp16"):` |
| Quantize region | **tta.quantize** | `with tta.quantize(mode="int8"):` |
| Annotate without editing | **tta.annotate** | `ann = tta.annotate(model); ann.autocast(where=..., mode="fp16"); tta.compile(model, inputs)` |
| Check region in CI | **tta.expect** | `with tta.expect("name", checks=[...]):` then `evaluate_all_regions(trt_model)` |
| Pick best impl | **tta.autotune** | `with tta.autotune(impls=[...], name="..."):` then `run_autotune(...)` |

---

### 13. Troubleshooting & limitations

**Region not lowered:** Use **require=True** to get a clear error, or **get_region_reports()** / **print_region_reports()** after compile for per-region status.

**RegionFailure:** Region couldn't be satisfied. Fix the impl or use `require=False` to fall back.

**TTADiagnosticError / TTAPluginError / TTABuiltinError:** Export/compile/runtime errors; check message and stage.

**SidecarConstraintError / SidecarSelectionError:** Invalid or conflicting **tta.annotate** (e.g. bad `where`).

**Limitations:** Blackwell + mixed Triton/CuTeDSL can hit CUDA 700 (use one backend per plugin). CuTile: cubin only, don't mix backends in one plugin. **tta.quantize:** module-centric; use `require=False`.

---

### 14. Running tests

Run inside the project dev Docker container (see repo `.cursorrules`).

```bash
python -m pytest tests/py/annotation/unit/ -q
python -m pytest tests/py/annotation/ -n 8
```

CuTile tests need Blackwell (sm_100+).
