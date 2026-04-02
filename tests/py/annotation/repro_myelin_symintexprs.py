#!/usr/bin/env python3
"""
Reproducer: Blackwell/Myelin QDP bug — onShapeChange fails when SymIntExprs is non-empty.

Root cause (revised)
--------------------
**Not a Myelin issue.**  Triton compiled against CUDA 13.x emits PTX 9.1.
TRT < 10.16 does not support PTX 9.1 and fails to JIT-compile the kernel; the
resulting ``onShapeChange`` failure at ``pluginUtilsRuntime.cpp`` is a symptom
of the kernel never loading, not a Myelin evaluator bug.

Fixed in TRT 10.16, which added PTX 9.1 support.  With a one-line downgrade
(``.version 9.1`` → ``.version 9.0``) the same plugins also work on TRT 10.16.

What this script demonstrates
------------------------------
BUG  repro::add_one_symint  — SymIntExprs(1), passes n_elements as a runtime
                               scalar arg to the Triton kernel.  FAILS on
                               Blackwell/Myelin: incorrect output.

CTRL repro::add_one_noarg   — SymIntExprs(0), n_elements is not needed (input
                               size guaranteed to be a multiple of BLOCK so no
                               masking is required).  PASSES on Blackwell/Myelin.

Both plugins implement the same logical operation (add 1.0 to every element).
The only difference is whether SymIntExprs is empty.

PTX compatibility notes
-----------------------
Triton on CUDA 13.x generates ``.version 9.1`` PTX targeting ``sm_120a``.
TRT's PTX JIT rejects PTX 9.1 but accepts PTX 9.0 with the same sm_120a target.
The ``_fix_ptx_for_trt`` helper in this script applies that single version
downgrade and also strips the extra internal params Triton appends to every
kernel entry (``printf_buffer``, ``prevGrid``, etc.) that TRT does not pass.

Verification result
-------------------
TRT 10.14.1: bug is present — onShapeChange fails on Blackwell for both
             SymIntExprs(0) and SymIntExprs(1) because PTX 9.1 is not accepted.
TRT 10.16.0: bug is FIXED — both variants pass after the PTX 9.0 downgrade.

Usage
-----
Run inside the Docker dev container with TRT 10.16:

    docker exec \\
        -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_libs:... \\
        -e CUDA_VISIBLE_DEVICES=2 \\
        torch_tensorrt_dev \\
        python /workspace/torch-trt/tests/py/annotation/repro_myelin_symintexprs.py
"""

import sys
from typing import Tuple, Optional, Union

try:
    import triton
    import triton.language as tl
    import triton.compiler
except ImportError as exc:
    print(f"ERROR: triton not available: {exc}")
    sys.exit(1)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp
except ImportError as exc:
    print(f"ERROR: TensorRT not available: {exc}")
    sys.exit(1)

try:
    import torch
    import numpy as np
except ImportError as exc:
    print(f"ERROR: torch/numpy not available: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK = 256   # Triton tile size (constexpr)
N = 1024      # Input length; must be a multiple of BLOCK for the noarg control


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _add_one_scalar_kernel(
    x_ptr,              # param 0: input pointer
    n_elements,         # param 1: runtime scalar → becomes SymIntExprs[0]
    y_ptr,              # param 2: output pointer
    BLOCK: tl.constexpr,
):
    """Elementwise add-1 with OOB masking via runtime n_elements scalar."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x + 1.0, mask=mask)


@triton.jit
def _add_one_noarg_kernel(
    x_ptr,              # param 0: input pointer
    y_ptr,              # param 1: output pointer
    BLOCK: tl.constexpr,
):
    """Elementwise add-1, no runtime scalar arg.  Safe only when N % BLOCK == 0."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x + 1.0)


def _fix_ptx_for_trt(ptx: str, kernel_name: str, num_used_params: int) -> str:
    """Two fixes needed for Triton PTX to work with TRT's QDP AOT path.

    1. PTX version downgrade: Triton with CUDA 13.x generates `.version 9.1`
       which TRT's PTX JIT cannot compile.  Downgrade to `.version 8.0` — the
       kernel only uses basic FP32 load/store/add so no PTX 9.x features are
       needed.  Remove the `a` suffix from `.target sm_120a` (TRT does not
       recognise the variant tag).

    2. Trailing-param strip: Triton appends extra internal params to every
       kernel (e.g. `printf_buffer`, `prevGrid`).  TRT passes exactly
       ``num_used_params`` arguments; extra `.param` declarations cause a
       CUDA_ERROR_LAUNCH_INVALID_CONFIG at runtime.  Strip all `.param` lines
       beyond the first ``num_used_params`` inside the kernel entry, then fix
       the trailing comma so the last kept param has none.
    """
    lines = ptx.split("\n")
    result = []
    pfx = f"{kernel_name}_param_"
    in_entry = False
    param_lines: list = []   # accumulate kept param lines
    pre_close_lines: list = []  # lines between last param and closing ')'

    for line in lines:
        # Fix 1: PTX version — Triton on CUDA 13.x emits .version 9.1 which
        # TRT 10.16's PTX JIT rejects.  Downgrade to 9.0.  The .target
        # sm_120a line is kept as-is; TRT handles it correctly.
        if line.startswith(".version"):
            result.append(".version 9.0")
            continue

        # Fix 2: collect and strip trailing params
        if f".entry {kernel_name}(" in line:
            in_entry = True
            param_lines = []
            result.append(line)
            continue

        if in_entry and ".param" in line and pfx in line:
            param_idx = len(param_lines)
            if param_idx < num_used_params:
                param_lines.append(line)
            continue

        if in_entry and ")" in line and ".param" not in line:
            in_entry = False
            # Emit kept params with correct commas (no trailing comma on last)
            for i, pline in enumerate(param_lines):
                pline = pline.rstrip().rstrip(",")
                if i < len(param_lines) - 1:
                    pline += ","
                result.append(pline)
            result.append(line)
            continue

        result.append(line)

    return "\n".join(result)


def _compile_kernel(fn, signature: dict, constexprs: dict):
    """Compile a Triton kernel and return (kernel_name, ptx_bytes, num_warps, shared_mem).

    Applies _fix_ptx_for_trt to make the PTX compatible with TRT's QDP runtime.
    """
    compiled = triton.compile(
        triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    )
    kernel_name: str = compiled.metadata.name
    ptx = compiled.asm["ptx"]
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8")
    num_used_params = len(signature)
    ptx = _fix_ptx_for_trt(ptx, kernel_name, num_used_params)
    return kernel_name, ptx.encode("utf-8"), compiled.metadata.num_warps, compiled.metadata.shared


# ---------------------------------------------------------------------------
# QDP registration — BUG variant: SymIntExprs(1)
# ---------------------------------------------------------------------------
# Param order in PTX: (x_ptr, n_elements, y_ptr)
# TRT runtime order: (inputs..., scalars..., outputs...) = (x_ptr, n_elements, y_ptr)
# → orders match; no PTX rewrite needed.

@trtp.register("repro::add_one_symint")
def _symint_desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()


@trtp.aot_impl("repro::add_one_symint")
def _symint_aot_impl(
    inp0: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymIntExprs]:

    kernel_name, ptx_bytes, num_warps, shared = _compile_kernel(
        fn=_add_one_scalar_kernel,
        signature={"x_ptr": "*fp32", "n_elements": "i32", "y_ptr": "*fp32"},
        constexprs={"BLOCK": BLOCK},
    )

    n = inp0.shape_expr[0]  # SymInt32 — symbolic input length

    launch = trtp.KernelLaunchParams()
    launch.grid_x = (n + (BLOCK - 1)) // BLOCK  # ceildiv, symbolic
    launch.grid_y = 1
    launch.grid_z = 1
    launch.block_x = num_warps * 32
    launch.block_y = 1
    launch.block_z = 1
    launch.shared_mem = shared

    # NON-EMPTY SymIntExprs: n_elements is passed as a scalar at runtime.
    # Myelin on Blackwell fails to evaluate this symbolic expression during
    # onShapeChange → incorrect output / engine failure.
    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = n

    return kernel_name.encode("utf-8"), ptx_bytes, launch, extra_args


# ---------------------------------------------------------------------------
# QDP registration — CTRL variant: SymIntExprs(0)
# ---------------------------------------------------------------------------

@trtp.register("repro::add_one_noarg")
def _noarg_desc(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.like()


@trtp.aot_impl("repro::add_one_noarg")
def _noarg_aot_impl(
    inp0: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymIntExprs]:

    kernel_name, ptx_bytes, num_warps, shared = _compile_kernel(
        fn=_add_one_noarg_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={"BLOCK": BLOCK},
    )

    n = inp0.shape_expr[0]  # SymInt32

    launch = trtp.KernelLaunchParams()
    launch.grid_x = (n + (BLOCK - 1)) // BLOCK
    launch.grid_y = 1
    launch.grid_z = 1
    launch.block_x = num_warps * 32
    launch.block_y = 1
    launch.block_z = 1
    launch.shared_mem = shared

    # EMPTY SymIntExprs: nothing for Myelin to evaluate → onShapeChange passes.
    extra_args = trtp.SymIntExprs(0)

    return kernel_name.encode("utf-8"), ptx_bytes, launch, extra_args


# ---------------------------------------------------------------------------
# Engine build + inference helper
# ---------------------------------------------------------------------------

_trt_logger = trt.Logger(trt.Logger.WARNING)


def build_and_run(
    ns: str,
    name: str,
    n: int,
) -> Tuple[bool, str, Optional[float]]:
    """Build a minimal TRT engine for ``ns::name``, run inference on ``n`` fp32 elements.

    Returns:
        (passed, message, max_abs_error)
    """
    builder = trt.Builder(_trt_logger)
    network = builder.create_network()
    config = builder.create_builder_config()

    x_inp = network.add_input("x", trt.float32, (n,))

    plugin_ns = getattr(trtp.op, ns)
    plugin_fn = getattr(plugin_ns, name)
    try:
        # plugin_fn(x_inp)(qpcr) → (input_tensors, shape_tensors, plugin_instance).
        # Pass STRICT_AOT so TRT uses the @trtp.aot_impl path (not @trtp.impl JIT).
        # add_plugin_v3 adds the layer to the network and returns IPluginV3Layer.
        inputs, shape_inputs, plugin = plugin_fn(x_inp)(trt.QuickPluginCreationRequest.STRICT_AOT)
        layer = network.add_plugin_v3(inputs, shape_inputs, plugin)
    except Exception as exc:
        return False, f"plugin layer creation failed: {exc}", None

    if layer is None:
        return False, "add_plugin_v3 returned None (aot_impl may have failed during registration)", None

    y_out = layer.get_output(0)
    network.mark_output(y_out)

    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except Exception as exc:
        return False, f"engine build raised: {exc}", None

    if engine_bytes is None:
        return False, "engine build returned None", None

    runtime = trt.Runtime(_trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        return False, "engine deserialization failed", None

    context = engine.create_execution_context()

    # Discover actual I/O names from the engine (don't assume fixed names).
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    if len(input_names) != 1 or len(output_names) != 1:
        return False, f"unexpected I/O count: inputs={input_names} outputs={output_names}", None

    x_host = np.ones(n, dtype=np.float32) * 2.0
    x_dev = torch.from_numpy(x_host).cuda()
    y_dev = torch.zeros(n, dtype=torch.float32).cuda()

    context.set_tensor_address(input_names[0], x_dev.data_ptr())
    context.set_tensor_address(output_names[0], y_dev.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    try:
        ok = context.execute_async_v3(stream)
    except Exception as exc:
        return False, f"execute_async_v3 raised: {exc}", None

    torch.cuda.synchronize()

    if not ok:
        return False, "execute_async_v3 returned False", None

    y_host = y_dev.cpu().numpy()
    expected = x_host + 1.0
    max_err = float(np.max(np.abs(y_host - expected)))
    if max_err > 1e-3:
        return False, f"wrong result: max_abs_err={max_err:.4f}, expected all {expected[0]:.1f}", max_err

    return True, f"max_abs_err={max_err:.2e}", max_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"TensorRT {trt.__version__}")
    dev = torch.cuda.get_device_properties(0)
    sm = f"sm_{dev.major}{dev.minor}"
    print(f"GPU     : {dev.name} ({sm})")
    is_blackwell = dev.major >= 10
    print(f"Myelin  : {'YES (Blackwell/sm_120+)' if is_blackwell else 'NO (pre-Blackwell)'}")
    print(f"N       : {N}  (BLOCK={BLOCK})")
    print()

    cases = [
        ("repro", "add_one_symint", "SymIntExprs(1)", "BUG "),
        ("repro", "add_one_noarg",  "SymIntExprs(0)", "CTRL"),
    ]

    results = {}
    for ns, name, tag, label in cases:
        full_name = f"{ns}::{name}"
        print(f"[{label}] {full_name}  [{tag}]", flush=True)
        passed, msg, _ = build_and_run(ns, name, N)
        status = "PASS" if passed else "FAIL"
        print(f"        → {status}: {msg}")
        results[full_name] = passed

    bug_pass = results["repro::add_one_symint"]
    ctrl_pass = results["repro::add_one_noarg"]

    print()
    print("=" * 60)
    if not bug_pass and ctrl_pass and is_blackwell:
        print("BUG CONFIRMED on Blackwell/Myelin:")
        print("  SymIntExprs(1) → FAIL  (onShapeChange cannot evaluate scalar exprs)")
        print("  SymIntExprs(0) → PASS  (nothing for Myelin to evaluate)")
        print()
        print("Known TRT bug on Blackwell. Fix expected in a future TRT release.")
        sys.exit(0)
    elif bug_pass and ctrl_pass:
        print("Both variants PASSED.")
        if is_blackwell:
            print("Bug appears to be FIXED in this TRT version!")
        else:
            print("(Pre-Blackwell: Myelin not used; expected behaviour.)")
        sys.exit(0)
    elif not bug_pass and not ctrl_pass:
        print("BOTH variants FAILED — likely an infrastructure issue (LD_LIBRARY_PATH?).")
        sys.exit(2)
    else:
        print(f"Unexpected result pattern (bug={bug_pass}, ctrl={ctrl_pass}).")
        sys.exit(3)


if __name__ == "__main__":
    main()
