"""Shared helpers for annotation E2E tests: compile/run + engine layer checks + accuracy vs eager."""

import ctypes
import json
import os
import subprocess
import sys
from typing import Optional

import tensorrt as trt
import torch

import torch_tensorrt
import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation._layer_metadata import parse_tta_layer_metadata
from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
    PythonTorchTensorRTModule,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    TorchTensorRTModule,
)


def _try_load_cudnn() -> bool:
    for soname in ("libcudnn.so.9", "libcudnn.so.8"):
        try:
            ctypes.CDLL(soname)
            return True
        except OSError:
            pass
    for p in getattr(sys, "path", []):
        if "site-packages" not in str(p):
            continue
        cudnn_lib = os.path.join(p, "nvidia", "cudnn", "lib")
        if not os.path.isdir(cudnn_lib):
            continue
        lp = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = cudnn_lib + (":" + lp if lp else "")
        for soname in ("libcudnn.so.9", "libcudnn.so.8"):
            path = os.path.join(cudnn_lib, soname)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path)
                    return True
                except OSError:
                    pass
    return False


def _install_cudnn() -> bool:
    pkgs = ["nvidia-cudnn-cu12<9"]
    cuda_major = getattr(torch.version, "cuda", "13") or "13"
    cuda_major = cuda_major.split(".")[0]
    if int(cuda_major) >= 13:
        pkgs.append(f"nvidia-cudnn-cu{cuda_major}")
    for pkg in pkgs:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pass
    import importlib
    importlib.invalidate_caches()
    return _try_load_cudnn()


def _has_cudnn8():
    if _try_load_cudnn():
        return True
    return _install_cudnn()


def tta_compile(model, inputs, **kwargs):
    """Test utility: torch_tensorrt.compile with TTA-friendly defaults."""
    kwargs.setdefault("min_block_size", 1)
    kwargs.setdefault("require_full_compilation", True)
    kwargs.setdefault("profiling_verbosity", trt.ProfilingVerbosity.DETAILED)
    return torch_tensorrt.compile(model, inputs=inputs, **kwargs)


def _compile_and_run_impl(model, compile_inputs, run_inputs, reference_out=None):
    model = model.eval().cuda()
    run_inputs = tuple(x.cuda() for x in run_inputs)
    if reference_out is not None:
        eager_out = reference_out if reference_out.is_cuda else reference_out.cuda()
    else:
        with torch.no_grad():
            eager_out = model(*run_inputs)
    trt_model = torch_tensorrt.compile(
        model,
        inputs=compile_inputs,
        min_block_size=1,
        require_full_compilation=True,
        profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
    )
    with torch.no_grad():
        trt_out = trt_model(*run_inputs)
    return trt_model, trt_out, eager_out


def _compile_and_run(model, inputs, reference_out=None):
    cuda_inputs = tuple(x.cuda() for x in inputs)
    return _compile_and_run_impl(model, cuda_inputs, cuda_inputs, reference_out)


def _compile_and_run_dynamic(model, trt_inputs, run_inputs, reference_out=None):
    """Compile with dynamic shape specs (torch_tensorrt.Input), run with concrete tensors.

    Args:
        model: nn.Module.
        trt_inputs: sequence of torch_tensorrt.Input or concrete tensors for build.
        run_inputs: sequence of concrete CUDA tensors to run inference with.
        reference_out: optional reference output tensor for accuracy comparison.
    """
    return _compile_and_run_impl(model, trt_inputs, run_inputs, reference_out)

_TRT_MODULE_TYPES = (PythonTorchTensorRTModule, TorchTensorRTModule)


def _is_trt_module(mod):
    return isinstance(mod, _TRT_MODULE_TYPES)


def _fix_metadata_value(raw: str) -> str:
    """TRT inspector can emit Metadata value as JSON with unescaped inner quotes. Escape them so outer JSON parses."""
    marker = '"Metadata": "'
    out = raw
    while True:
        start = out.find(marker)
        if start == -1:
            break
        value_start = start + len(marker)
        if value_start >= len(out) or out[value_start] != "{":
            break
        end_markers = ('}}"\n', '}}"\r', '}}"\n}', '}}" ', '}}"')
        end_quote = -1
        for em in end_markers:
            idx = out.find(em, value_start)
            if idx != -1:
                end_quote = idx
                break
        if end_quote == -1:
            idx = out.find('}}"', value_start)
            if idx != -1:
                end_quote = idx
        if end_quote == -1:
            break
        content = out[value_start:end_quote].replace("\\", "\\\\").replace('"', '\\"')
        out = out[:value_start] + content + out[end_quote:]
    return out


def _get_engine_layer_info_raw(mod) -> str:
    """Return engine layer info JSON using an execution context for full format strings.

    TRT's engine inspector reports abbreviated "Format/Datatype" values (e.g.
    "Half") when no execution context is set.  With an execution context the
    inspector emits the full verbose description (e.g. "Channel major FP16
    format where channel % 8 == 0") which uniquely identifies non-LINEAR
    tensor formats like HWC8.

    Uses the module's existing execution context (``mod.context``) when
    available to avoid creating a duplicate context.  On architectures where
    ``get_engine_information`` fails with an execution context (e.g. Blackwell
    Myelin engines) falls back to the no-context call.
    """
    engine = getattr(mod, "engine", None)
    if engine is not None:
        ctx = getattr(mod, "context", None)
        try:
            inspector = engine.create_engine_inspector()
            if ctx is not None:
                inspector.execution_context = ctx
            raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            if raw:
                return raw
        except Exception:
            pass
    # Fallback: use the module's own get_layer_info() without execution context.
    return mod.get_layer_info()


def get_trt_engine_layer_info(trt_model):
    layers = []
    for _name, mod in trt_model.named_modules():
        if _is_trt_module(mod):
            raw = _get_engine_layer_info_raw(mod)
            try:
                info = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    info = json.loads(raw.replace("\n", "").replace("\r", ""))
                except json.JSONDecodeError:
                    info = json.loads(_fix_metadata_value(raw))
            for layer in info.get("Layers", []):
                if isinstance(layer, dict):
                    layers.append(dict(layer))
                else:
                    layers.append({"Name": str(layer)})
    return layers


def get_trt_engine_layer_names(trt_model):
    return [layer.get("Name", "") for layer in get_trt_engine_layer_info(trt_model)]


def get_tta_metadata_from_layer(layer_dict):
    """Parse TTA metadata from a layer dict. Inspector getLayerInformation(getEngineInformation)(kJSON) returns JSON with 'Name', 'Metadata', 'LayerType', ... per layer; 'Metadata' is populated only when the engine was built with ProfilingVerbosity.DETAILED and setMetadata() was called for that layer."""
    raw = layer_dict.get("Metadata") or layer_dict.get("metadata") or ""
    if not raw:
        return None
    tta_meta = parse_tta_layer_metadata(raw)
    if tta_meta is not None:
        return tta_meta
    try:
        data = json.loads(raw)
        return data.get("tta")
    except (json.JSONDecodeError, TypeError):
        return None


def get_layers_with_tta_metadata(trt_model):
    out = []
    for layer in get_trt_engine_layer_info(trt_model):
        tta_meta = get_tta_metadata_from_layer(layer)
        if tta_meta is not None:
            out.append((layer, tta_meta))
    return out



def count_trt_engines(trt_model):
    return sum(1 for _name, mod in trt_model.named_modules() if _is_trt_module(mod))


def _tta_meta_matches(meta, expected):
    if not meta:
        return False
    fn_specs = meta.get("fn_specs", [])
    for key, value in expected.items():
        if key == "fn_name":
            # Check that any fn_spec entry has this fn_name.
            if not any(e.get("fn_name") == value for e in fn_specs):
                return False
        elif key == "fn_config":
            # Check that any fn_spec entry has all expected config key=value pairs.
            if not any(
                all(e.get("config", {}).get(k) == v for k, v in value.items())
                for e in fn_specs
            ):
                return False
        elif meta.get(key) != value:
            return False
    return True


def get_selected_tactic_for_engine(trt_model):
    """Return {plugin_name: {"idx": tactic_id}} for each TTA plugin layer in the engine.

    Reads TacticValue (hex tactic index) from the engine layer JSON and
    correlates it with the plugin_name from TTA metadata.
    Both are available with ProfilingVerbosity.DETAILED.
    """
    result = {}
    for layer in get_trt_engine_layer_info(trt_model):
        # TacticValue is present on pre-Blackwell (e.g. A100); Blackwell uses
        # TacticName (empty string for QDP plugins).  Only populate result when
        # TacticValue is available.
        tactic_hex = layer.get("TacticValue", "")
        meta = get_tta_metadata_from_layer(layer)
        if meta is None or not tactic_hex:
            continue
        plugin_name = meta.get("plugin_name", "")
        if not plugin_name:
            continue
        tactic_id = int(tactic_hex, 16)
        result[plugin_name] = {"idx": tactic_id}
    return result


def assert_tactics_metadata(test_case, trt_model, expected_tactics):
    """Assert that the engine has TTA layers with metadata.

    The ``tactics`` field was removed from the metadata format; this function
    now just verifies that TTA-annotated layers exist in the compiled engine.
    The ``expected_tactics`` argument is accepted for backwards compatibility
    but is no longer used for matching (tactics encoding was removed).
    """
    with_tta = get_layers_with_tta_metadata(trt_model)
    test_case.assertGreater(len(with_tta), 0, msg="No TTA layer metadata found")


# TRT's engine inspector (with execution_context) emits verbose format descriptions.
# Map them to short canonical names used in assertions.
_TRT_FORMAT_ALIASES = [
    ("channel % 8 == 0",  "HWC8"),
    ("channel % 16 == 0", "HWC16"),
    ("Row major linear FP32",    "Float"),
    ("Row major linear FP16",    "Half"),
    ("Row major linear BF16",    "BF16"),
    ("Row major linear INT8",    "Int8"),
    ("Row major linear INT32",   "Int32"),
]


def _normalize_trt_format(fmt: str) -> str:
    """Normalize TRT's verbose format/datatype string to a short canonical name."""
    for pattern, alias in _TRT_FORMAT_ALIASES:
        if pattern in fmt:
            return alias
    return fmt


def get_plugin_io_formats(trt_model):
    """Return the I/O Format/Datatype strings TRT actually used for each TTA plugin layer.

    Returns a list of dicts with keys ``layer_name``, ``input_formats``, ``output_formats``
    for every plugin layer that carries TTA metadata.  The format strings are
    normalized from TRT's verbose descriptions (e.g. "Channel major FP16 format
    where channel % 8 == 0") to short canonical names (e.g. "HWC8").
    """
    result = []
    for layer in get_trt_engine_layer_info(trt_model):
        meta = get_tta_metadata_from_layer(layer)
        if meta is None:
            continue
        inp_fmts = [
            _normalize_trt_format(i.get("Format/Datatype", ""))
            for i in layer.get("Inputs", [])
        ]
        out_fmts = [
            _normalize_trt_format(o.get("Format/Datatype", ""))
            for o in layer.get("Outputs", [])
        ]
        result.append({
            "layer_name": layer.get("Name", ""),
            "input_formats": inp_fmts,
            "output_formats": out_fmts,
        })
    return result


# Mapping from declared format substring to the dtype Myelin reports for it.
# Myelin (Blackwell) fuses layers and only exposes the dtype token (e.g. "Half"),
# not the layout (e.g. "HWC8").  These aliases let the format assert degrade
# gracefully: when Myelin is detected we check the dtype instead.
_MYELIN_FORMAT_TO_DTYPE = {
    "HWC8":  "Half",
    "HWC16": "Half",
    "CHW16": "Half",
    "CHW4":  "Int8",
    "CHW32": "Int8",
    "Half":  "Half",
    "Float": "Float",
    "BF16":  "BF16",
    "Int8":  "Int8",
    "Int32": "Int32",
}


def assert_plugin_io_format(test_case, trt_model, expected_format_substr):
    """Assert that at least one TTA plugin layer has I/O format matching *expected_format_substr*.

    TRT's JSON inspector reports ``"Format/Datatype"`` per tensor, e.g. ``"Float"``,
    ``"HWC8 | FP16"``, ``"CHW4 | Int8"``.  Pass a substring such as ``"HWC8"`` or
    ``"Float"`` to check that TRT actually selected the requested format.

    On Blackwell GPUs, TRT uses Myelin compilation which fuses layers and reports
    only the dtype (not layout) in the engine inspector.  When Myelin layers are
    detected, the assertion falls back to checking the dtype token that corresponds
    to the requested format (e.g. "HWC8" → "Half", "Float" → "Float").
    """
    formats = get_plugin_io_formats(trt_model)
    test_case.assertGreater(len(formats), 0, "No TTA plugin layers found in engine")
    # Detect Myelin: all TTA layer names contain "_myl" (Myelin fusion marker).
    myelin = all("_myl" in e["layer_name"] for e in formats)
    check_substr = _MYELIN_FORMAT_TO_DTYPE.get(expected_format_substr, expected_format_substr) if myelin else expected_format_substr
    for entry in formats:
        all_fmts = entry["input_formats"] + entry["output_formats"]
        if any(check_substr in f for f in all_fmts):
            return
    actual = [(e["input_formats"], e["output_formats"]) for e in formats]
    test_case.fail(
        f"Expected format containing {expected_format_substr!r}"
        + (f" (Myelin dtype: {check_substr!r})" if myelin else "")
        + f" not found in any TTA plugin layer I/O; got {actual}"
    )


def assert_trt_compiled(
    test_case,
    trt_model,
    trt_out,
    eager_out,
    atol=1e-3,
    rtol=1e-3,
    expected_tta_metadata=None,
    allow_dtype_mismatch=False,
):
    if allow_dtype_mismatch:
        torch.testing.assert_close(
            trt_out.float() if trt_out.dtype != torch.float32 else trt_out,
            eager_out.float() if eager_out.dtype != torch.float32 else eager_out,
            atol=atol,
            rtol=rtol,
        )
    else:
        torch.testing.assert_close(trt_out, eager_out, atol=atol, rtol=rtol)
    test_case.assertEqual(count_trt_engines(trt_model), 1)
    layer_info = get_trt_engine_layer_info(trt_model)
    test_case.assertGreater(len(layer_info), 0)
    if expected_tta_metadata:
        with_tta = get_layers_with_tta_metadata(trt_model)
        test_case.assertGreater(
            len(with_tta),
            0,
            msg="No layers with TTA metadata (engine must be built with ProfilingVerbosity.DETAILED)",
        )
        for expected_meta in expected_tta_metadata:
            found = any(
                _tta_meta_matches(meta, expected_meta) for _layer, meta in with_tta
            )
            test_case.assertTrue(
                found,
                f"Expected TTA metadata {expected_meta!r} in layer metadata; got {[(m.get('backend'), m.get('plugin_name')) for _, m in with_tta]}",
            )


def assert_region_reports(
    test_case,
    ep,
    expected: list,
):
    """Assert that get_region_reports(ep) matches expected specifications.

    expected is a list of dicts, one per expected report, in region_id order:
        [{"kind": "autocast", "effect": "applied"}, {"kind": "quantize", "effect": "applied"}, ...]

    Checks:
    - len(reports) == len(expected)
    - reports[i].kind == expected[i]["kind"]
    - reports[i].effect == expected[i]["effect"]
    - If expected[i] has "mode" key: reports[i].details["mode"] == expected[i]["mode"]
    - If expected[i] has "cast_ops_inserted_gt" key: reports[i].details["cast_ops_inserted"] > expected[i]["cast_ops_inserted_gt"]
    """
    from torch_tensorrt.annotation._compile.pipeline import get_region_reports
    reports = get_region_reports(ep)
    test_case.assertEqual(
        len(reports), len(expected),
        f"Expected {len(expected)} RegionReport(s), got {len(reports)}: "
        f"{[(r.kind, r.effect) for r in reports]}"
    )
    for i, (report, exp) in enumerate(zip(reports, expected)):
        test_case.assertEqual(
            report.kind, exp["kind"],
            f"report[{i}].kind: expected {exp['kind']!r}, got {report.kind!r}"
        )
        test_case.assertEqual(
            report.effect, exp["effect"],
            f"report[{i}] ({report.kind}:{report.region_id}): "
            f"expected effect={exp['effect']!r}, got {report.effect!r}"
        )
        if "mode" in exp:
            test_case.assertEqual(
                report.details.get("mode"), exp["mode"],
                f"report[{i}].details['mode']: expected {exp['mode']!r}"
            )
        if "cast_ops_inserted_gt" in exp:
            test_case.assertGreater(
                report.details.get("cast_ops_inserted", 0),
                exp["cast_ops_inserted_gt"],
                f"report[{i}].details['cast_ops_inserted'] should be > {exp['cast_ops_inserted_gt']}"
            )
