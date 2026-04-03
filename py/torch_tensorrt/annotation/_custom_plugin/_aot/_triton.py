"""Triton AOT backend: source → sandbox recording → PTX → KernelLaunchParams.

AOT compilation pipeline for Triton kernels
============================================
1. **Sandbox execution** — The user's ``launch_fn`` is run with ``SymbolicTensor``
   proxies in place of real tensors and ``TritonLaunchRecorder`` objects injected
   over the real ``@triton.jit`` kernels in the module globals.  This lets us
   capture the kernel call without executing on a GPU.

2. **Argument analysis** — ``analyze_launch_args`` separates the recorded call
   arguments into *pointer binding indices* (which TRT tensor buffer maps to
   which kernel parameter) and *scalar SymInt32 expressions* (grid or shape
   scalars that TRT evaluates at runtime).

3. **PTX compilation** — ``triton.compile(ASTSource(fn, signature, constexprs))``
   produces both PTX and CUBIN.  We extract the PTX from ``compiled.asm["ptx"]``
   because TRT's QDP runtime JIT-compiles PTX for the current GPU architecture,
   matching the official ``aot_plugin`` example and avoiding cubin arch mismatch.

4. **Parameter reordering** — Triton emits `.param` declarations in Python/kernel
   order (pointers first, then scalars).  The QDP runtime passes arguments in
   *runtime order* (input pointers, scalars, output pointers).  ``_fix_triton_ptx_for_trt``
   rewrites the `.param` block and all references in the PTX body to match.

5. **Tactic uniquification** — When multiple tactics compile the same kernel
   function with different ``constexprs`` (e.g. ``BLOCK_M=16`` vs ``BLOCK_M=32``),
   a config-derived suffix is appended to the kernel name so TRT registers
   separate PTX entries per tactic.

The public entry-point for the descriptor system is ``aot_impl_triton``, which
returns the QDP AOT 4-tuple ``(kernel_name, ptx_bytes, KernelLaunchParams, SymIntExprs)``.
``compile_triton_kernel`` is a thin wrapper used by the tactic manager.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp
except ImportError as e:
    raise ImportError(
        "TensorRT with plugin support is required for Triton AOT compilation."
    ) from e

from .._qdp_utils import (
    AOTMetadata,
    TTAPluginError,
    _as_symint32,
    _launch_params_from_trt,
    analyze_launch_args,
    dump_code_artifact,
    is_triton_kernel,
    run_kernel_sandbox,
)
from ..._recorders import TritonLaunchRecorder
from ..._specs import TritonSpec
from .._symbolic import SymbolicTensor, TensorRole


def _trt_dtype_to_triton_ptr(trt_dtype: Any, qdp_symbol: str) -> str:
    """Map a TensorRT DataType to a Triton pointer element-type string."""
    if trt_dtype == trt.float16:
        return "fp16"
    if trt_dtype == trt.bfloat16:
        return "bf16"
    if trt_dtype == trt.float32:
        return "fp32"
    if trt_dtype == trt.int32:
        return "i32"
    raise TTAPluginError(
        op=qdp_symbol,
        stage="aot_impl",
        backend="triton",
        msg=f"unsupported tensor dtype {trt_dtype} for Triton kernel signature",
    )


# LIMITATION (fragile PTX rewrite): _fix_triton_ptx_for_trt rewrites .param
# declarations and their references inside the PTX body using line-by-line text
# scanning.  Triton compiles parameters in Python/kernel order (pointers first,
# then constexprs), while the TRT plugin runtime expects (inputs, scalars,
# outputs).  The rewrite is fragile because:
#   1. It identifies param references by matching the prefix ``{kernel_name}_param_``
#      as a plain string — any change to Triton's param naming convention silently
#      produces incorrect PTX.
#   2. It scans for ``.entry {kernel_name}(`` as a literal string; multi-line or
#      differently-formatted entry declarations will not be recognised.
#   3. Unused trailing params (constexprs not referenced in the body) are stripped
#      by counting references — this is correct only if Triton doesn't reuse param
#      indices in non-obvious ways.
# The proper fix is for TRT's QDP AOT API to expose a parameter-order remapping
# mechanism so that post-compilation PTX rewriting is not needed.
def _fix_triton_ptx_for_trt(
    ptx: str,
    kernel_name: str,
    num_used_params: int,
    param_binding_indices: List[int],
    num_inputs: int,
    num_scalars: int,
) -> str:
    """Reorder PTX params, strip unused trailing params, and downgrade PTX version.

    Three rewrites applied:

    1. **PTX version downgrade**: Triton on CUDA 13.x emits ``.version 9.1``.
       TRT's QDP PTX JIT caps at 9.0 — kernels with 9.1 silently fail to load,
       producing a spurious ``onShapeChange`` error at runtime.  Downgrade the
       ``.version`` line to 9.0.  The ``.target`` line (e.g. ``sm_120a``) is
       left unchanged.

    2. **Param reorder**: Triton compiles with params in Python/kernel order
       (ptrs first, then scalars).  The plugin runtime passes arguments in
       (inputs, scalars, outputs) order.  Reorder ``.param`` declarations and
       their body references to match.

    3. **Trailing-param strip**: Triton appends internal params (``printf_buffer``,
       ``prevGrid``) beyond the user-declared arguments.  TRT passes exactly
       ``num_used_params`` args; strip the extras and fix the trailing comma.
    """
    # Rewrite 1: PTX version downgrade (9.x → 9.0).
    # Triton on CUDA 13.x emits `.version 9.1`; TRT's QDP PTX JIT caps at 9.0.
    # Only touch lines where the major version is 9 and the minor is > 0.
    lines = ptx.split("\n")
    result_pre: List[str] = []
    for line in lines:
        if line.startswith(".version "):
            line = re.sub(
                r"^(\.version\s+)9\.([1-9]\d*)",
                r"\g<1>9.0",
                line,
            )
        result_pre.append(line)
    ptx = "\n".join(result_pre)

    # Rewrites 2 & 3: param reorder + trailing-param strip.
    num_ptrs = len(param_binding_indices)
    input_params = []
    output_params = []
    for orig_idx, binding in enumerate(param_binding_indices):
        if binding < num_inputs:
            input_params.append((binding, orig_idx))
        else:
            output_params.append((binding, orig_idx))
    input_params.sort()
    output_params.sort()
    scalar_params = list(range(num_ptrs, num_ptrs + num_scalars))
    trt_order = (
        [orig for _, orig in input_params]
        + scalar_params
        + [orig for _, orig in output_params]
    )
    needs_reorder = trt_order != list(range(num_used_params))
    pfx = f"{kernel_name}_param_"
    lines = ptx.split("\n")
    result: List[str] = []
    in_entry = False
    param_lines: List[str] = []

    for line in lines:
        if f".entry {kernel_name}(" in line:
            in_entry = True
            param_lines = []
            result.append(line)
            continue

        if in_entry and ".param" in line and f"{pfx}" in line:
            param_lines.append(line)
            continue

        if in_entry and ")" in line and ".param" not in line:
            in_entry = False
            if needs_reorder:
                reordered = [param_lines[i] for i in trt_order if i < len(param_lines)]
            else:
                reordered = param_lines[:num_used_params]
            for i, pline in enumerate(reordered):
                pline = pline.rstrip().rstrip(",")
                if i < len(reordered) - 1:
                    pline += ","
                result.append(pline)
            result.append(line)
            continue

        result.append(line)

    if needs_reorder:
        old_to_new = {old: new for new, old in enumerate(trt_order)}
        joined = "\n".join(result)
        width = len(str(num_used_params - 1)) if num_used_params > 0 else 1
        for old_idx in range(num_used_params - 1, -1, -1):
            joined = joined.replace(
                f"{pfx}{old_idx}", f"{pfx}TEMP{old_idx:0{width}d}"
            )
        for old_idx, new_idx in old_to_new.items():
            joined = joined.replace(
                f"{pfx}TEMP{old_idx:0{width}d}", f"{pfx}{new_idx}"
            )
        result = joined.split("\n")

    return "\n".join(result)


def aot_impl_triton(
    *,
    qdp_symbol: str,
    spec: TritonSpec,
    cfg: Mapping[str, Any],
    launch_fn: Any,
    host_args: List[SymbolicTensor],
    inp_descs: List[Any],
    out_descs: List[Any],
    attrs: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, bytes, Any, Any]:
    """Triton AOT implementation: sandbox → record → compile → PTX.

    Steps:
    1. Find @triton.jit kernels in launch_fn's module.
    2. Replace with TritonLaunchRecorder proxies.
    3. Run sandboxed launch_fn(*host_args, **merged_kwargs).
    4. Analyse recorded args → param_binding_indices + scalar SymInts.
    5. Build per-tensor-dtype Triton signature, triton.compile() → PTX.
    6. Fix PTX: reorder params to runtime order, strip unused trailing params.
    7. Return (kernel_name, ptx, KernelLaunchParams, SymIntExprs).

    Returns:
        (kernel_name, ptx, KernelLaunchParams, SymIntExprs)
    """
    import triton

    backend = "triton"

    merged_kwargs = dict(cfg)
    if attrs:
        merged_kwargs.update(attrs)

    # 1-2. Locate module, find @triton.jit kernels, sandbox and run.
    #      strict=True: propagate module-not-found and no-kernel-found as errors.
    #      host_kwargs: cfg (tactic constexprs) + attrs (plugin compile-time constants).
    # Broad catch is necessary: the sandbox executes arbitrary user launch_fn code
    # and can raise any exception type (ImportError for missing deps, TypeError
    # for shape mismatches, AttributeError from proxy gaps, etc.).
    try:
        used_recorder, kernel_recorders = run_kernel_sandbox(
            launch_fn=launch_fn,
            host_args=host_args,
            is_kernel_fn=is_triton_kernel,
            recorder_factory=lambda obj: TritonLaunchRecorder(real_kernel=obj),
            host_kwargs=merged_kwargs,
            strict=True,
            op=qdp_symbol,
            backend=backend,
        )
    except TTAPluginError:
        raise
    except Exception as exc:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"sandbox failed for op '{qdp_symbol}': {exc}",
        ) from exc

    # 3. Exactly one Triton kernel must have been launched.
    used = [rec for rec in kernel_recorders.values() if rec.grid is not None]
    if len(used) != 1:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"expected exactly 1 Triton kernel launch; got {len(used)}",
        )

    recorder = used[0]
    grid = recorder.grid
    args = recorder.args
    kwargs = recorder.kwargs or {}

    num_inputs = len(inp_descs)
    num_outputs = len(out_descs)

    def _to_int(x: Any) -> Any:
        try:
            return int(x)
        except (TypeError, ValueError):
            return x

    constexprs = {k: _to_int(v) for k, v in cfg.items()}
    if attrs:
        constexprs.update(attrs)
    kernel_arg_names = list(recorder.real_kernel.arg_names)
    positional_names = [n for n in kernel_arg_names if n not in constexprs]

    full_args: List[Any] = []
    for i, name in enumerate(positional_names):
        if i < len(args):
            full_args.append(args[i])
        elif name in kwargs:
            full_args.append(kwargs[name])
        else:
            raise TTAPluginError(
                op=qdp_symbol,
                stage="aot_impl",
                backend=backend,
                msg=f"missing kernel argument '{name}' (pass positionally or by keyword)",
            )

    # 4. Analyze recorded args → pointer binding indices + scalar SymInts.
    param_binding_indices, scalar_symints = analyze_launch_args(
        args=full_args,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        op=qdp_symbol,
        backend=backend,
    )

    if not param_binding_indices:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg="no pointer arguments recorded for Triton kernel",
        )

    # 5. Build Triton signature dict {param_name: type_str} for non-constexpr args.
    all_descs = list(inp_descs) + list(out_descs)

    ptr_idx = 0
    scalar_idx = 0
    signature: Dict[str, str] = {}
    for name in positional_names:
        if ptr_idx < len(param_binding_indices):
            b = param_binding_indices[ptr_idx]
            dtype_str = _trt_dtype_to_triton_ptr(all_descs[b].dtype, qdp_symbol)
            signature[name] = f"*{dtype_str}"
            ptr_idx += 1
        else:
            signature[name] = "i32"
            scalar_idx += 1

    expected_num_scalars = len(positional_names) - len(param_binding_indices)
    if len(scalar_symints) != expected_num_scalars:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=(
                f"scalar count mismatch: launch passed {len(scalar_symints)} scalars "
                f"but kernel has {expected_num_scalars} scalar args (depends on descriptor ranks for this invocation)"
            ),
        )

    # triton.compile raises a mix of Exception subclasses (CompilationError,
    # subprocess.CalledProcessError, RuntimeError) depending on the failure mode;
    # a broad catch is necessary here to give a structured diagnostic in all cases.
    try:
        compiled = triton.compile(
            triton.compiler.ASTSource(
                fn=recorder.real_kernel,
                signature=signature,
                constexprs=constexprs,
            )
        )
    except Exception as exc:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"triton.compile failed for kernel '{recorder.real_kernel.__name__}': {exc}",
        ) from exc

    # 6. Build KernelLaunchParams (grid can be symbolic; QDP evaluates at runtime).
    # Pattern: grid from shape-derived expressions, block/shared from compiled metadata,
    # extra_args from shape_expr-derived scalars; only tile sizes in constexprs.
    launch = trtp.KernelLaunchParams()
    if isinstance(grid, tuple):
        launch.grid_x = _as_symint32(grid[0]) if len(grid) >= 1 else trtp.SymInt32(1)
        launch.grid_y = _as_symint32(grid[1]) if len(grid) >= 2 else trtp.SymInt32(1)
        launch.grid_z = _as_symint32(grid[2]) if len(grid) >= 3 else trtp.SymInt32(1)
    else:
        launch.grid_x = _as_symint32(grid)
        launch.grid_y = trtp.SymInt32(1)
        launch.grid_z = trtp.SymInt32(1)

    launch.block_x = compiled.metadata.num_warps * 32
    launch.block_y = 1
    launch.block_z = 1
    launch.shared_mem = compiled.metadata.shared

    extra_args = trtp.SymIntExprs(len(scalar_symints))
    for idx, val in enumerate(scalar_symints):
        extra_args[idx] = _as_symint32(val)

    # 7. Extract PTX, reorder params to runtime order, strip unused trailing params.
    kernel_name_str: str = compiled.metadata.name
    ptx: str = compiled.asm["ptx"]
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8")

    dump_code_artifact("TTA_DUMP_TRITON_PTX", f"{kernel_name_str}_raw.ptx", ptx)

    # Triton 3.x names kernels after the Python function (e.g. "_add_2d_kernel"),
    # without any per-constexpr hash.  When two tactics compile the same function
    # with different constexprs (e.g. BLOCK_M=16 vs BLOCK_M=32), both return the
    # same kernel_name.  TRT identifies kernels by name, so it uses whichever PTX
    # was registered last for *all* tactics sharing that name — but still applies
    # each tactic's launch params, causing a mismatch (wrong grid dimensions for
    # the baked-in tile sizes).  Append a short config suffix so every tactic gets
    # a distinct name.
    if cfg:
        suffix = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
        unique_name = f"{kernel_name_str}_{suffix}"
        ptx = ptx.replace(kernel_name_str, unique_name)
        kernel_name_str = unique_name

    num_used = len(positional_names)
    ptx = _fix_triton_ptx_for_trt(
        ptx=ptx,
        kernel_name=kernel_name_str,
        num_used_params=num_used,
        param_binding_indices=param_binding_indices,
        num_inputs=num_inputs,
        num_scalars=len(scalar_symints),
    )
    ptx_bytes = ptx.encode("utf-8")

    dump_code_artifact("TTA_DUMP_TRITON_PTX", f"{kernel_name_str}_fixed.ptx", ptx)

    return kernel_name_str, ptx_bytes, launch, extra_args


def compile_triton_kernel(spec: TritonSpec, config: Dict[str, Any]) -> AOTMetadata:
    """Compile a TritonSpec into unified AOTMetadata.

    This is the tactic-manager entry-point.  It constructs synthetic 1-input /
    1-output TensorDesc stubs (shape [256], float32) to drive the sandbox run,
    delegates to ``aot_impl_triton`` for the full pipeline, and wraps the result
    in ``AOTMetadata``.

    Args:
        spec:   TritonSpec carrying ``launch_fn``, optional ``configs``, etc.
        config: Single tactic configuration dict (e.g. ``{"BLOCK_M": 32}``).

    Returns:
        AOTMetadata with ``backend="triton"``, compiled PTX bytes, and launch params.
    """
    inp_descs = [trtp.TensorDesc(dtype=trt.float32, shape_expr=[256])]
    out_descs = [trtp.TensorDesc(dtype=trt.float32, shape_expr=[256])]
    sym_inputs = [
        SymbolicTensor(td=inp_descs[0], role=TensorRole.INPUT, index=0)
    ]
    sym_outputs = [
        SymbolicTensor(td=out_descs[0], role=TensorRole.OUTPUT, index=0)
    ]
    host_args = sym_inputs + sym_outputs
    kernel_name_str, ptx_bytes, launch, extra_args = aot_impl_triton(
        qdp_symbol="triton_compile",
        spec=spec,
        cfg=config,
        launch_fn=spec.launch_fn,
        host_args=host_args,
        inp_descs=inp_descs,
        out_descs=out_descs,
    )
    launch_params = _launch_params_from_trt(launch, extra_args, num_inputs=1, num_outputs=1)
    return AOTMetadata(binary=ptx_bytes, kernel_name=kernel_name_str, launch_params=launch_params, backend="triton")
