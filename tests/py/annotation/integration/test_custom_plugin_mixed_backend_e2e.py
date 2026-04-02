"""E2E: custom_plugin with mixed backend specs (Triton + CuTeDSL).

A single custom_plugin([spec, ...]) creates a tactic table with one entry per
(spec, config) combination.  TRT autotune picks the fastest tactic.  These
tests verify that all combinations compile, that TRT executes without error,
and that the numeric output matches eager.

HARDWARE: These tests run on a pre-Blackwell GPU (e.g. A100, sm_80).
All test classes are marked ``@pytest.mark.requires_pre_bw``; conftest.py
automatically reruns them in a subprocess with a pre-Blackwell GPU when
Blackwell GPUs are present.

WHY pre-Blackwell only — TensorRT 10.14 bug on Blackwell (Myelin QUICKAOT):
  When a QDP plugin's tactics return SymIntExprs of different lengths, TRT/Myelin
  crashes with CUDA error 700 (illegal memory access) at execute_async_v3.
  Mixed Triton+CuTeDSL plugins trigger this:
    - Triton tactics return SymIntExprs(N) where N = number of scalar kernel args
      (e.g. n_elements); the values are evaluated from tensor shape at runtime and
      passed as extra int32 arguments to the PTX kernel.
    - CuTeDSL tactics return SymIntExprs(0); tensor shape is carried by the CuTe
      tensor descriptor, so no extra scalar args are needed.
  On Blackwell, TRT/Myelin selects one SymIntExprs length for the whole plugin and
  applies it when dispatching any tactic — passing extra args to a kernel whose PTX
  has no corresponding .param declarations → illegal memory access.
  On pre-Blackwell (non-Myelin path), the mismatch is tolerated silently.
  Filed as a TensorRT bug.  Repro: tests/py/annotation/repro_blackwell_extra_len_mismatch.py

NOTE: CuTile is NOT included here.  CuTile 1.1 only supports sm_100+ (Blackwell)
and generates a cubin (not PTX), making it incompatible with mixed-backend plugins
on any architecture.  Single-backend CuTile tests live in
test_custom_plugin_cutile_e2e.py and run on Blackwell via conftest.py.
"""

import unittest

import pytest
import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
import torch_tensorrt.annotation as tta

from ._e2e_common import _compile_and_run, _compile_and_run_dynamic, assert_plugin_io_format, assert_tactics_metadata, assert_trt_compiled, get_selected_tactic_for_engine
from .cutedsl_kernels import (
    launch_add_one as cutedsl_launch_add_one,
    launch_conv3x3_relu_pool as cutedsl_launch_conv3x3_relu_pool,
    launch_fma as cutedsl_launch_fma,
    launch_fused_add_relu as cutedsl_launch_fused_add_relu,
    launch_scale as cutedsl_launch_scale,
)
# CuTile kernels are imported for reference; CuTile is excluded from mixed-backend
# tests because CuTile 1.1 only supports sm_100+ (Blackwell) and its cubin calling
# convention is incompatible with Triton/CuTeDSL PTX in mixed plugins on Blackwell.
from .cutile_kernels import (
    launch_conv3x3_relu_pool as cutile_launch_conv3x3_relu_pool,
)
from .triton_kernels import (
    launch_add_one as triton_launch_add_one,
    launch_conv3x3_relu_pool as triton_launch_conv3x3_relu_pool,
    launch_fma as triton_launch_fma,
    launch_fused_add_relu as triton_launch_fused_add_relu,
    launch_scale as triton_launch_scale,
)


# ---------------------------------------------------------------------------
# TestMixedBackendE2E (Triton + CuTile) — removed.
# CuTile 1.1 only supports sm_100+ (Blackwell) and produces a cubin rather
# than PTX.  On Blackwell, TRT 10.14 stores one cubin slot per plugin, so the
# last-registered cubin overwrites earlier ones and is used for ALL tactics at
# runtime.  When CuTile is mixed with Triton in the same plugin, TRT always
# runs CuTile's kernel regardless of the selected tactic, breaking Triton
# tactics whose calling convention differs from CuTile's.
# Single-backend CuTile tests live in test_custom_plugin_cutile_e2e.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Triton + CuTeDSL
# ---------------------------------------------------------------------------


@pytest.mark.requires_pre_bw
class TestMixedBackendTritonCuTeDSLE2E(unittest.TestCase):
    """Custom plugin with Triton + CuTeDSL backend variants.

    Both Triton and CuTeDSL produce PTX; TRT JIT-compiles each tactic's PTX
    independently at runtime.  Spec order: [cutedsl, triton].
    """

    def test_add_one_cutedsl_and_triton(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_scale_cutedsl_and_triton(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_scale, configs=[{}]),
                tta.triton(triton_launch_scale, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def scale(x):
                return x * 2.0

            def forward(self, x):
                return self.scale(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_fma_cutedsl_and_triton(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_fma, configs=[{}]),
                tta.triton(triton_launch_fma, configs=[{"BLOCK_SIZE": 64}]),
            ]))
            def fma_op(a, b, c):
                return a * b + c

            def forward(self, a, b, c):
                return self.fma_op(a, b, c)

        inputs = (torch.randn(64), torch.randn(64), torch.randn(64))
        trt_model, trt_out, eager_out = _compile_and_run(M(), inputs)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_fused_add_relu_cutedsl_and_triton(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_fused_add_relu, configs=[{}]),
                tta.triton(triton_launch_fused_add_relu, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def fused_op(x, y):
                return torch.relu(x + y)

            def forward(self, x, y):
                return self.fused_op(x, y)

        inputs = (torch.randn(128), torch.randn(128))
        trt_model, trt_out, eager_out = _compile_and_run(M(), inputs)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_three_tactics_one_cutedsl_two_triton(self):
        """Three tactics: CuTeDSL, Triton BLOCK=128, Triton BLOCK=256."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(triton_launch_add_one, configs=[
                    {"BLOCK_SIZE": 128},
                    {"BLOCK_SIZE": 256},
                ]),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_chained_mixed_ops_triton_cutedsl(self):
        """Chain two ops; each has CuTeDSL and Triton tactics."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 64}]),
            ]))
            def add_one(x):
                return x + 1.0

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_scale, configs=[{}]),
                tta.triton(triton_launch_scale, configs=[{"BLOCK_SIZE": 64}]),
            ]))
            def scale(x):
                return x * 2.0

            def forward(self, x):
                return self.scale(self.add_one(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(32),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_dynamic_shape_triton_cutedsl(self):
        """Triton+CuTeDSL with dynamic 1D batch profile."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        trt_inputs = [
            torch_tensorrt.Input(
                min_shape=(64,), opt_shape=(128,), max_shape=(256,),
                dtype=torch.float32,
            )
        ]
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(
            M(), trt_inputs, (torch.randn(128, device="cuda"),)
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_tactics_metadata_triton_cutedsl(self):
        """Verify tactics metadata lists CuTeDSL and Triton backend entries."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}]),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(self, trt_model, trt_out, eager_out)
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "cutedsl", "fn_name": "launch_add_one", "config": {}},
            {"idx": 2, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 64}},
            {"idx": 3, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 128}},
        ])


# TestMixedBackendCuTileCuTeDSLE2E removed — CuTile excluded from mixed-backend tests.
# See module docstring. Single-backend CuTile tests: test_custom_plugin_cutile_e2e.py.

# TestMixedBackendFormatCuTileCuTeDSLE2E removed — same reason as above.

# TestThreeBackendE2E removed — included CuTile, excluded for same reason as above.

# TestMixedBackendFormatE2E removed — all tests used Triton+CuTile mixed backend.


# ---------------------------------------------------------------------------
# Triton + CuTeDSL: format variants
# ---------------------------------------------------------------------------


@pytest.mark.requires_pre_bw
class TestMixedBackendFormatTritonCuTeDSLE2E(unittest.TestCase):
    """Mixed Triton + CuTeDSL backend with explicit format declarations."""

    def test_triton_explicit_format_cutedsl_default(self):
        """Triton spec with explicit LINEAR; CuTeDSL spec with default format."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(
                    triton_launch_add_one,
                    configs=[{"BLOCK_SIZE": 128}],
                    input_formats=[trt.TensorFormat.LINEAR],
                    output_formats=[trt.TensorFormat.LINEAR],
                ),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "cutedsl", "fn_name": "launch_add_one"},
            {"idx": 2, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 128}},
        ])

    def test_both_specs_explicit_linear_format(self):
        """Both CuTeDSL and Triton specs explicitly declare LINEAR format."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(
                    cutedsl_launch_scale,
                    configs=[{}],
                    input_formats=[trt.TensorFormat.LINEAR],
                    output_formats=[trt.TensorFormat.LINEAR],
                ),
                tta.triton(
                    triton_launch_scale,
                    configs=[{"BLOCK_SIZE": 64}],
                    input_formats=[trt.TensorFormat.LINEAR],
                    output_formats=[trt.TensorFormat.LINEAR],
                ),
            ]))
            def scale(x):
                return x * 2.0

            def forward(self, x):
                return self.scale(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "cutedsl", "fn_name": "launch_scale"},
            {"idx": 2, "backend": "triton", "fn_name": "launch_scale"},
        ])

    def test_triton_cutedsl_hwc8(self):
        """Both CuTeDSL and Triton specs declare HWC8 format.

        Shape (1, 8, 4, 4): batch=1, C=8 (divisible by 8), H=4, W=4.
        With a 4D tensor satisfying HWC8 constraints, TRT selects HWC8 for
        this layer.  The test verifies that the format declaration is forwarded
        to AutoTuneCombination and that TRT honours it.
        """
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(
                    cutedsl_launch_add_one,
                    configs=[{}],
                    input_formats=[trt.TensorFormat.HWC8],
                    output_formats=[trt.TensorFormat.HWC8],
                ),
                tta.triton(
                    triton_launch_add_one,
                    configs=[{"BLOCK_SIZE": 128}],
                    input_formats=[trt.TensorFormat.HWC8],
                    output_formats=[trt.TensorFormat.HWC8],
                ),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.ones(1, 8, 4, 4, dtype=torch.float16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_plugin_io_format(self, trt_model, "HWC8")

    def test_triton_cutedsl_fma_hwc8(self):
        """FMA with CuTeDSL and Triton specs both declaring HWC8 format.

        Shape (1, 8, 4, 4) for all three inputs: batch=1, C=8 (divisible by 8),
        H=4, W=4.  With 4D tensors satisfying HWC8 constraints, TRT selects
        HWC8 for this layer.  Uses torch.ones for simple FMA numerics.
        """
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(
                    cutedsl_launch_fma,
                    configs=[{}],
                    input_formats=[trt.TensorFormat.HWC8],
                    output_formats=[trt.TensorFormat.HWC8],
                ),
                tta.triton(
                    triton_launch_fma,
                    configs=[{"BLOCK_SIZE": 64}],
                    input_formats=[trt.TensorFormat.HWC8],
                    output_formats=[trt.TensorFormat.HWC8],
                ),
            ]))
            def fma_op(a, b, c):
                return a * b + c

            def forward(self, a, b, c):
                return self.fma_op(a, b, c)

        inputs = (torch.ones(1, 8, 4, 4, dtype=torch.float16), torch.ones(1, 8, 4, 4, dtype=torch.float16), torch.ones(1, 8, 4, 4, dtype=torch.float16))
        trt_model, trt_out, eager_out = _compile_and_run(M(), inputs)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_plugin_io_format(self, trt_model, "HWC8")

    def test_triton_hwc8_4d(self):
        """Triton add_one with HWC8 format declared on a ≥3D tensor.

        Shape (1, 8, 4, 4): batch=1, C=8 (divisible by 8), H=4, W=4.
        With a 4D tensor satisfying HWC8 constraints, and with the spec's
        input_formats forwarded to AutoTuneCombination, TRT selects HWC8 for
        this layer.  The test verifies:
          1. The plugin compiles and executes correctly on the 4D tensor.
          2. The format declaration round-trips (spec stores HWC8 in metadata).
          3. TRT selects HWC8 as the IO format.
        """
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.triton(
                    triton_launch_add_one,
                    configs=[{"BLOCK_SIZE": 128}],
                    input_formats=[trt.TensorFormat.HWC8],
                    output_formats=[trt.TensorFormat.HWC8],
                ),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        x = torch.ones(1, 8, 4, 4, dtype=torch.float16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_plugin_io_format(self, trt_model, "HWC8")


# ---------------------------------------------------------------------------
# nn.Conv2d + ReLU + 2x2 max pool: Triton backend
# ---------------------------------------------------------------------------


@pytest.mark.requires_pre_bw
class TestMixedBackendConvReluPoolE2E(unittest.TestCase):
    """nn.Conv2d + ReLU + pool with Triton backend tactics.

    Demonstrates that nn.Conv2d weights (.weight, .bias) flow correctly as
    constant plugin inputs when passed explicitly to the annotated static method.
    """

    @staticmethod
    def _meta(x, weight, bias):
        out = torch.nn.functional.conv2d(x, weight, bias, padding=1)
        return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

    def test_conv3x3_relu_pool_nn_conv2d_mixed(self):
        """nn.Conv2d module; instance method uses self.conv(x) with Triton tactics."""
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
                nn.init.normal_(self.conv.weight, std=0.1)
                nn.init.zeros_(self.conv.bias)

            @tta.export_as(impl=tta.custom_plugin([
                tta.triton(triton_launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 4}]),
                # CuTile excluded — requires sm_100+: tta.cutile(cutile_launch_conv3x3_relu_pool, configs=[{}]),
            ], meta_impl=meta))
            def conv_relu_pool(self, x,
                               weight=tta.self_attr("conv.weight"),
                               bias=tta.self_attr("conv.bias")):
                out = self.conv(x)
                return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

            def forward(self, x):
                return self.conv_relu_pool(x)

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_conv3x3_relu_pool_nn_conv2d_no_bias_mixed(self):
        """nn.Conv2d without bias; zero bias buffer via self_attr."""
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
                nn.init.normal_(self.conv.weight, std=0.1)
                self.register_buffer("zero_bias", torch.zeros(8))

            @tta.export_as(impl=tta.custom_plugin([
                tta.triton(triton_launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 8}]),
                # CuTile excluded — requires sm_100+: tta.cutile(cutile_launch_conv3x3_relu_pool, configs=[{}]),
            ], meta_impl=meta))
            def conv_relu_pool(self, x,
                               weight=tta.self_attr("conv.weight"),
                               bias=tta.self_attr("zero_bias")):
                out = self.conv(x)
                return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

            def forward(self, x):
                return self.conv_relu_pool(x)

        x = torch.randn(1, 4, 16, 16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_conv3x3_relu_pool_two_triton_configs(self):
        """Three tactics: Triton BLOCK_C=4, Triton BLOCK_C=8."""
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
                nn.init.normal_(self.conv.weight, std=0.1)
                nn.init.zeros_(self.conv.bias)

            @tta.export_as(impl=tta.custom_plugin([
                tta.triton(triton_launch_conv3x3_relu_pool, configs=[
                    {"BLOCK_C": 4},
                    {"BLOCK_C": 8},
                ]),
                # CuTile excluded — requires sm_100+: tta.cutile(cutile_launch_conv3x3_relu_pool, configs=[{}]),
            ], meta_impl=meta))
            def conv_relu_pool(self, x,
                               weight=tta.self_attr("conv.weight"),
                               bias=tta.self_attr("conv.bias")):
                out = self.conv(x)
                return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

            def forward(self, x):
                return self.conv_relu_pool(x)

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_conv3x3_relu_pool_cutedsl_and_triton_self_attr(self):
        """CuTeDSL + Triton mixed tactics with self_attr weights (nn.Conv2d.weight/.bias).

        Verifies that self_attr weight/bias tensors flow correctly through both the
        CuTeDSL AOT path and the Triton AOT path when they share the same plugin.
        """
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
                nn.init.normal_(self.conv.weight, std=0.1)
                nn.init.zeros_(self.conv.bias)

            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_conv3x3_relu_pool, configs=[{}]),
                tta.triton(triton_launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 4}]),
            ], meta_impl=meta))
            def conv_relu_pool(self, x,
                               weight=tta.self_attr("conv.weight"),
                               bias=tta.self_attr("conv.bias")):
                out = self.conv(x)
                return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

            def forward(self, x):
                return self.conv_relu_pool(x)

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )


# ---------------------------------------------------------------------------
# Autotune tactic selection: verify non-last tactic can win
# ---------------------------------------------------------------------------


@pytest.mark.requires_pre_bw
class TestAutotuneNonLastTacticE2E(unittest.TestCase):
    """Verify TRT's autotuner can select a non-last (first) tactic.

    Uses two Triton configs where tactic 1 (BLOCK_SIZE=1024) is far faster
    than tactic 2 (BLOCK_SIZE=4) for a large tensor:
      BLOCK_SIZE=1024  ->  4096 blocks for 4M elements  (efficient)
      BLOCK_SIZE=4     ->  1M blocks for 4M elements     (massive launch overhead)

    On pre-Blackwell (A100), TRT compiles each tactic's PTX independently and
    benchmarks both; TacticValue in the engine JSON identifies the winner.
    """

    def test_first_tactic_selected_over_last(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 1024}]),
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 4}]),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        x = torch.ones(4 * 1024 * 1024)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(self, trt_model, trt_out, eager_out,
                            expected_tta_metadata=[{"backend": "triton"}])

        selected = get_selected_tactic_for_engine(trt_model)
        self.assertTrue(selected, "No TacticValue in engine JSON — pre-Blackwell GPU required")
        spec_id = next(iter(selected))
        tactic = selected[spec_id]
        self.assertEqual(
            tactic.get("idx"), 1,
            f"Expected tactic idx=1 (BLOCK_SIZE=1024) to win over idx=2 (BLOCK_SIZE=4), "
            f"got: {tactic}",
        )


# ---------------------------------------------------------------------------
# Mixed-backend dtype tests: FP16 and BF16 with Triton + CuTeDSL
# ---------------------------------------------------------------------------


@pytest.mark.requires_pre_bw
class TestMixedBackendDtypeE2E(unittest.TestCase):
    """FP16 and BF16 inputs through mixed Triton + CuTeDSL custom plugins."""

    def test_fp16_add_one_triton_cutedsl(self):
        """Mixed Triton + CuTeDSL add_one with float16 inputs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        x = torch.randn(256, dtype=torch.float16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_bf16_add_one_triton_cutedsl(self):
        """Mixed Triton + CuTeDSL add_one with bfloat16 inputs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        x = torch.randn(256, dtype=torch.bfloat16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=3e-2, rtol=3e-2,  # bf16 element-wise: max 1 ULP ≈ 3e-2
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_fp16_scale_triton_cutedsl(self):
        """Mixed Triton + CuTeDSL scale with float16 inputs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_scale, configs=[{}]),
                tta.triton(triton_launch_scale, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def scale(x):
                return x * 2.0

            def forward(self, x):
                return self.scale(x)

        x = torch.randn(256, dtype=torch.float16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_bf16_scale_triton_cutedsl(self):
        """Mixed Triton + CuTeDSL scale with bfloat16 inputs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.cutedsl(cutedsl_launch_scale, configs=[{}]),
                tta.triton(triton_launch_scale, configs=[{"BLOCK_SIZE": 128}]),
            ]))
            def scale(x):
                return x * 2.0

            def forward(self, x):
                return self.scale(x)

        x = torch.randn(256, dtype=torch.bfloat16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=3e-2, rtol=3e-2,  # bf16 element-wise: max 1 ULP ≈ 3e-2
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )


# ---------------------------------------------------------------------------
# lower_as with multi-spec custom_plugin (single region, Triton + CuTeDSL)
# ---------------------------------------------------------------------------


@pytest.mark.requires_pre_bw
class TestLowerAsMixedSingleRegionE2E(unittest.TestCase):
    """lower_as with a single region backed by a multi-spec (Triton + CuTeDSL) custom_plugin.

    A single lower_as region with impl=tta.custom_plugin([spec, ...]) registers
    multiple backend tactics for that region.  TRT autotune picks the fastest.
    Runs on A100 (pre-Blackwell) in this file's subprocess.
    """

    def test_lower_as_triton_cutedsl_add_one(self):
        """lower_as region backed by CuTeDSL + Triton add_one tactics."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin([
                        tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                        tta.triton(triton_launch_add_one, configs=[{"BLOCK_SIZE": 128}]),
                    ]),
                    require=True, name="mixed_add_one",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_lower_as_triton_cutedsl_scale(self):
        """lower_as region backed by CuTeDSL + Triton scale tactics."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin([
                        tta.cutedsl(cutedsl_launch_scale, configs=[{}]),
                        tta.triton(triton_launch_scale, configs=[{"BLOCK_SIZE": 128}]),
                    ]),
                    require=True, name="mixed_scale",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )

    def test_lower_as_triton_cutedsl_three_tactics(self):
        """lower_as region with CuTeDSL + two Triton BLOCK_SIZE configs (three tactics total)."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin([
                        tta.cutedsl(cutedsl_launch_add_one, configs=[{}]),
                        tta.triton(triton_launch_add_one, configs=[
                            {"BLOCK_SIZE": 64},
                            {"BLOCK_SIZE": 128},
                        ]),
                    ]),
                    require=True, name="mixed_add_one_three",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl"}],
        )
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "cutedsl", "fn_name": "launch_add_one", "config": {}},
            {"idx": 2, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 64}},
            {"idx": 3, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 128}},
        ])
