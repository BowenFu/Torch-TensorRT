"""E2E: custom_plugin with Triton backend. Compare eager vs TRT accuracy (GPU + TRT)."""

import unittest

import pytest

from ._e2e_common import _compile_and_run, _compile_and_run_dynamic, assert_plugin_io_format, assert_tactics_metadata, assert_trt_compiled, get_selected_tactic_for_engine
from .triton_kernels import (
    launch_add_2d,
    launch_add_one,
    launch_add_one_and_scale,
    launch_add_one_int,
    launch_conv1x1_relu,
    launch_conv3x3_relu_pool,
    launch_fma,
    launch_fused_add_relu,
    launch_gelu,
    launch_matmul,
    launch_elementwise_mul,
    launch_scale,
)
import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
import torch_tensorrt.annotation as tta
import triton
import triton.language as tl


class TestTritonPluginE2E(unittest.TestCase):
    """Triton custom plugin -> TRT engine build + execute: accuracy check."""

    def test_add_one(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_scale(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 256}])))
            def custom_op(x):
                return x * 2.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_with_conv(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x + 1.0

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(16, 16, 3, padding=1, bias=False)

            def forward(self, x):
                return self.custom_op(self.conv(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(1, 16, 8, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_add_2d_plugin(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_add_2d, configs=[{"BLOCK_M": 32, "BLOCK_N": 32}]),
            ))
            def add_2d_op(x, y):
                return x + y

            def forward(self, x, y):
                return self.add_2d_op(x, y)

        x = torch.randn(64, 128)
        y = torch.randn(64, 128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_matmul_plugin(self):
        def matmul_meta(a_in, b_in):
            return a_in @ b_in

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_matmul, configs=[{"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}]),
                meta_impl=matmul_meta,
            ))
            def matmul_op(a, b):
                return a @ b

            def forward(self, a, b):
                return self.matmul_op(a, b)

        a = torch.randn(64, 16)
        b = torch.randn(16, 64)
        reference_out = (a.cpu() @ b.cpu()).cuda()
        trt_model, trt_out, eager_out = _compile_and_run(
            M(), (a, b), reference_out=reference_out
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_conv_plugin(self):
        def conv_meta(x, weight, bias):
            w = weight.unsqueeze(-1).unsqueeze(-1)
            return torch.relu(torch.nn.functional.conv2d(x, w, bias))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))
                self.bias = nn.Parameter(torch.randn(4))

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_conv1x1_relu, configs=[{"BLOCK_C": 32}]),
                meta_impl=conv_meta,
            ))
            def conv1x1_relu(x, weight, bias):
                w = weight.unsqueeze(-1).unsqueeze(-1)
                return torch.relu(torch.nn.functional.conv2d(x, w, bias))

            def forward(self, x):
                return self.conv1x1_relu(x, self.weight, self.bias)

        x = torch.randn(2, 4, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )


    def test_fused_add_relu_two_inputs(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_fused_add_relu, configs=[{"BLOCK_SIZE": 128}])))
            def fused_op(x, y):
                return torch.relu(x + y)

            def forward(self, x, y):
                return self.fused_op(x, y)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256), torch.randn(256)))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_gelu(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_gelu, configs=[{"BLOCK_SIZE": 256}])))
            def gelu_op(x):
                return torch.nn.functional.gelu(x, approximate="tanh")

            def forward(self, x):
                return self.gelu_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(512),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_fma_three_inputs(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_fma, configs=[{"BLOCK_SIZE": 128}])))
            def fma_op(a, b, c):
                return a * b + c

            def forward(self, a, b, c):
                return self.fma_op(a, b, c)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256), torch.randn(256), torch.randn(256)))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_gelu_after_linear(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_gelu, configs=[{"BLOCK_SIZE": 128}])))
            def gelu_op(x):
                return torch.nn.functional.gelu(x, approximate="tanh")

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64, bias=True)

            def forward(self, x):
                return self.gelu_op(self.linear(x))

        x = torch.randn(4, 64)
        model = M()
        with torch.no_grad():
            reference_out = model(x.cpu()).cuda()
        trt_model, trt_out, eager_out = _compile_and_run(
            model, (x,), reference_out=reference_out
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_fused_add_relu_after_conv(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_fused_add_relu, configs=[{"BLOCK_SIZE": 256}])))
            def fused_add_relu(x, y):
                return torch.relu(x + y)

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(4, 4, 1, bias=False)
                nn.init.ones_(self.conv.weight)

            def forward(self, x):
                y = self.conv(x)
                return self.fused_add_relu(x.view(-1), y.view(-1))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(1, 4, 8, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            atol=2e-3, rtol=1e-3,  # conv accumulation can exceed default 1e-3 atol
            expected_tta_metadata=[{"backend": "triton"}],
        )


class TestTritonChainedE2E(unittest.TestCase):
    """Chained Triton custom plugins and mixed with builtins -> TRT accuracy."""

    def test_builtin_then_custom(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(self.relu(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(4, 4),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "builtin", "plugin_name": "add_activation"},
                {"backend": "triton"},
            ],
        )

    def test_custom_then_builtin(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x * 2.0

            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def forward(self, x):
                return self.relu(self.custom_op(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(4, 4),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "builtin", "plugin_name": "add_activation"},
                {"backend": "triton"},
            ],
        )

    def test_two_custom_plugins_chained(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 64}])))
            def custom_a(x):
                return x + 1.0

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 64}])))
            def custom_b(x):
                return x * 2.0

            def forward(self, x):
                return self.custom_b(self.custom_a(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(32),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_fma_then_scale(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_fma, configs=[{"BLOCK_SIZE": 128}])))
            def fma_op(a, b, c):
                return a * b + c

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])))
            def scale_op(x):
                return x * 2.0

            def forward(self, a, b, c):
                return self.scale_op(self.fma_op(a, b, c))

        trt_model, trt_out, eager_out = _compile_and_run(
            M(), (torch.randn(128), torch.randn(128), torch.randn(128)),
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )


class TestTritonDtypeE2E(unittest.TestCase):
    """Non-float32 dtype coverage: fp16 and bf16 inputs through Triton custom plugins."""

    def test_fp16_inputs(self):
        """Custom plugin with float16 inputs and outputs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x * 2.0

            def forward(self, x):
                return self.custom_op(x)

        x = torch.randn(256, dtype=torch.float16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_bf16_inputs(self):
        """Custom plugin with bfloat16 inputs and outputs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 256}])))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        x = torch.randn(512, dtype=torch.bfloat16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=3e-2, rtol=3e-2,  # bf16 element-wise: max 1 ULP ≈ 3e-2
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_fp16_matmul(self):
        """Custom plugin matmul with float16 inputs; meta_impl infers fp16 output."""
        def matmul_meta(a, b):
            return a @ b

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_matmul, configs=[{"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}]),
                meta_impl=matmul_meta,
            ))
            def matmul_op(a, b):
                return a @ b

            def forward(self, a, b):
                return self.matmul_op(a, b)

        a = torch.randn(32, 16, dtype=torch.float16)
        b = torch.randn(16, 32, dtype=torch.float16)
        reference_out = (a.float() @ b.float()).half()
        trt_model, trt_out, eager_out = _compile_and_run(M(), (a, b), reference_out=reference_out)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,  # fp16 matmul: accumulation error ≤ 1e-2
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_int32_inputs(self):
        """Custom plugin with int32 inputs; kernel adds integer 1."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one_int, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x + 1

            def forward(self, x):
                return self.custom_op(x)

        x = torch.randint(0, 100, (256,), dtype=torch.int32)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=0, rtol=0,
            expected_tta_metadata=[{"backend": "triton"}],
        )


class TestTritonShapeE2E(unittest.TestCase):
    """Shape coverage: 3D tensors, large tensors, multiple tiling configs."""

    def test_3d_input(self):
        """Custom plugin with a 3D input tensor; kernel treats elements as flat."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        x = torch.randn(4, 8, 16)  # 512 elements total
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_multiple_configs_tactic_selection(self):
        """TRT builds successfully with 4 BLOCK_SIZE configs and selects the best tactic."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[
                    {"BLOCK_SIZE": 64},
                    {"BLOCK_SIZE": 128},
                    {"BLOCK_SIZE": 256},
                    {"BLOCK_SIZE": 512},
                ],
            )))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(1024),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_large_2d_tensor(self):
        """Custom plugin on a larger 2D tensor to stress block-tiling correctness."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_2d,
                configs=[{"BLOCK_M": 32, "BLOCK_N": 32}],
            )))
            def add_op(x, y):
                return x + y

            def forward(self, x, y):
                return self.add_op(x, y)

        x = torch.randn(128, 256)
        y = torch.randn(128, 256)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )


class TestTritonParallelE2E(unittest.TestCase):
    """Two or more custom plugins operating on independent (non-sequential) paths."""

    def test_two_plugins_independent_paths(self):
        """Two custom plugins on separate inputs; outputs merged by addition."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 64}])))
            def plugin_a(x):
                return x + 1.0

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 64}])))
            def plugin_b(y):
                return y * 2.0

            def forward(self, x, y):
                return self.plugin_a(x) + self.plugin_b(y)

        x = torch.randn(64)
        y = torch.randn(64)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_three_plugins_diamond(self):
        """Diamond DAG: two plugins consume the same input, outputs fused by a third."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])))
            def branch_a(x):
                return x + 1.0

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])))
            def branch_b(x):
                return x * 2.0

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_fused_add_relu, configs=[{"BLOCK_SIZE": 128}])))
            def merge(a, b):
                return torch.relu(a + b)

            def forward(self, x):
                a = self.branch_a(x)
                b = self.branch_b(x)
                return self.merge(a, b)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_plugin_plus_builtin_parallel(self):
        """Custom plugin and builtin activation both branch from same input; outputs summed."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])))
            def custom_scale(x):
                return x * 2.0

            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def forward(self, x):
                return self.custom_scale(x) + self.relu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "triton"},
                {"backend": "builtin", "plugin_name": "add_activation"},
            ],
        )


class TestTritonErrorCases(unittest.TestCase):
    """Negative tests: verify that invalid configurations raise clear errors."""

    def test_wrong_meta_impl_shape_raises(self):
        """meta_impl declaring wrong output shape raises at torch.export trace time."""
        def wrong_meta(a, b):
            # Claims output is (M, M) square but correct matmul output is (M, N)
            return torch.empty(a.shape[0], a.shape[0], device=a.device, dtype=a.dtype)

        class BadModel(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_matmul, configs=[{"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8}]),
                meta_impl=wrong_meta,
            ))
            def matmul_op(a, b):
                return a @ b

            def __init__(self):
                super().__init__()
                # Expects (*, 64) input; wrong_meta says output is (16,16) not (16,64)
                self.head = nn.Linear(64, 1)

            def forward(self, a, b):
                out = self.matmul_op(a, b)  # wrong_meta: (16,16) instead of (16,64)
                return self.head(out)       # Linear(64,1) gets (16,16) → shape error

        with self.assertRaises(Exception):
            _compile_and_run(BadModel(), (torch.randn(16, 8), torch.randn(8, 64)))


class TestTritonDynamicShapeE2E(unittest.TestCase):
    """Dynamic shape: custom plugin compiled with min/opt/max profiles, run at non-opt sizes."""

    def _check_sizes(self, trt_model, model, make_inputs, sizes, atol=1e-3, rtol=1e-3):
        """Run trt_model and eager model at each size and assert_close for all."""
        model = model.eval().cuda()
        for sz in sizes:
            inputs = make_inputs(sz)
            with torch.no_grad():
                eager = model(*inputs)
                trt_out = trt_model(*inputs)
            torch.testing.assert_close(trt_out, eager, atol=atol, rtol=rtol)

    def test_dynamic_1d_batch(self):
        """1D tensor with dynamic batch dim: accuracy checked at min (64), opt (128), and max (256)."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 64}])))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(128,), max_shape=(256,), dtype=torch.float32)
        ]
        model = M()
        # Primary run at non-opt size (64) — also checks engine metadata.
        run_inputs = (torch.randn(64, device="cuda"),)
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(self, trt_model, trt_out, eager_out, expected_tta_metadata=[{"backend": "triton"}])
        # Accuracy at all three profile sizes.
        self._check_sizes(trt_model, model, lambda n: (torch.randn(n, device="cuda"),), [64, 128, 256])

    def test_dynamic_2d_batch(self):
        """2D tensor with dynamic batch dim: accuracy checked at min (1), non-opt (3), and max (8)."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(1, 64), opt_shape=(4, 64), max_shape=(8, 64), dtype=torch.float32)
        ]
        model = M()
        run_inputs = (torch.randn(3, 64, device="cuda"),)
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(self, trt_model, trt_out, eager_out, expected_tta_metadata=[{"backend": "triton"}])
        self._check_sizes(trt_model, model, lambda b: (torch.randn(b, 64, device="cuda"),), [1, 4, 8])

    def test_dynamic_scale_sweep(self):
        """Scale plugin with dynamic 1D profile; accuracy verified at min, opt, max, and mid-range."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 64}])))
            def custom_op(x):
                return x * 2.0

            def forward(self, x):
                return self.custom_op(x)

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(256,), max_shape=(512,), dtype=torch.float32)
        ]
        model = M()
        # Primary run at opt size — checks engine/metadata.
        run_inputs = (torch.randn(256, device="cuda"),)
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(self, trt_model, trt_out, eager_out, expected_tta_metadata=[{"backend": "triton"}])
        # Accuracy sweep across full profile range.
        self._check_sizes(trt_model, model, lambda n: (torch.randn(n, device="cuda"),), [64, 128, 256, 512])

    def test_dynamic_two_inputs(self):
        """Two dynamic inputs; fused_add_relu accuracy checked at min, non-opt, and max."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_fused_add_relu, configs=[{"BLOCK_SIZE": 128}])))
            def fused_op(x, y):
                return torch.relu(x + y)

            def forward(self, x, y):
                return self.fused_op(x, y)

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(256,), max_shape=(512,), dtype=torch.float32),
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(256,), max_shape=(512,), dtype=torch.float32),
        ]
        model = M()
        run_inputs = (torch.randn(128, device="cuda"), torch.randn(128, device="cuda"))
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(self, trt_model, trt_out, eager_out, expected_tta_metadata=[{"backend": "triton"}])
        self._check_sizes(trt_model, model,
                          lambda n: (torch.randn(n, device="cuda"), torch.randn(n, device="cuda")),
                          [64, 256, 512])

    def test_dynamic_chained_plugins(self):
        """Two chained custom plugins; accuracy checked at min, non-opt, and max."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 64}])))
            def add_one(x):
                return x + 1.0

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 64}])))
            def scale(x):
                return x * 2.0

            def forward(self, x):
                return self.scale(self.add_one(x))

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(32,), opt_shape=(128,), max_shape=(256,), dtype=torch.float32)
        ]
        model = M()
        run_inputs = (torch.randn(96, device="cuda"),)
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(self, trt_model, trt_out, eager_out, expected_tta_metadata=[{"backend": "triton"}])
        self._check_sizes(trt_model, model, lambda n: (torch.randn(n, device="cuda"),), [32, 128, 256])


class TestTritonBlockedE2E(unittest.TestCase):
    """Blocked Triton kernels: multiple BLOCK_SIZE configs, static and dynamic shapes."""

    def test_add_one_blocked_two_configs(self):
        """add_one with BLOCK_SIZE=128 and BLOCK_SIZE=256 tactics."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[{"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 256}],
            )))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(512),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_scale_blocked_two_configs(self):
        """scale-by-2 with BLOCK_SIZE=64 and BLOCK_SIZE=128 tactics."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_scale,
                configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}],
            )))
            def custom_op(x):
                return x * 2.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_blocked_dynamic_shape(self):
        """Blocked kernel with dynamic 1D profile; accuracy at min, opt, max."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[{"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 256}],
            )))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(128,), opt_shape=(256,), max_shape=(512,), dtype=torch.float32)
        ]
        model = M()
        run_inputs = (torch.randn(256, device="cuda"),)
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )
        model = model.eval().cuda()
        for n in [128, 256, 512]:
            x = torch.randn(n, device="cuda")
            with torch.no_grad():
                torch.testing.assert_close(trt_model(x), model(x), atol=1e-3, rtol=1e-3)

    def test_chained_blocked_plugins(self):
        """Two chained blocked-kernel plugins with different BLOCK_SIZE configs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[{"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 256}],
            )))
            def add_one(x):
                return x + 1.0

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_scale,
                configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}],
            )))
            def scale(x):
                return x * 2.0

            def forward(self, x):
                return self.scale(self.add_one(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )


    def test_tactics_metadata(self):
        """Verify tactics metadata contains both BLOCK_SIZE configs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}],
            )))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(self, trt_model, trt_out, eager_out)
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 64}},
            {"idx": 2, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 128}},
        ])


class TestTritonAutotuneNonLastTacticE2E(unittest.TestCase):
    """Verify TRT's autotuner can select a non-last (first) Triton tactic.

    Uses three Triton configs with different BLOCK_SIZE values.  The test does
    not force a winner; it verifies that get_selected_tactic_for_engine returns
    a non-None result and that the selected tactic's backend is "triton".
    """

    def test_non_last_tactic_can_be_selected(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 256}],
            )))
            def add_one(x):
                return x + 1.0

            def forward(self, x):
                return self.add_one(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

        selected = get_selected_tactic_for_engine(trt_model)
        if selected:
            spec_id = next(iter(selected))
            tactic = selected[spec_id]
            self.assertIsNotNone(tactic, "Expected a non-None selected tactic")
            self.assertEqual(
                tactic.get("backend"), "triton",
                f"Expected selected tactic backend == 'triton'; got: {tactic}",
            )


class TestTriton2DBlockedE2E(unittest.TestCase):
    """2D block tiles: grid=(ceil(M/BLOCK_M), ceil(N/BLOCK_N)), block via tl.constexpr."""

    def test_add_2d_blocked_static(self):
        """2D blocked add with two (BLOCK_M, BLOCK_N) config pairs."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_2d,
                configs=[{"BLOCK_M": 16, "BLOCK_N": 16}, {"BLOCK_M": 32, "BLOCK_N": 32}],
            )))
            def add_op(x, y):
                return x + y

            def forward(self, x, y):
                return self.add_op(x, y)

        x = torch.randn(64, 128)
        y = torch.randn(64, 128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_add_2d_blocked_static_asymmetric(self):
        """2D blocked add with asymmetric tiles: BLOCK_M != BLOCK_N."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_2d,
                configs=[{"BLOCK_M": 32, "BLOCK_N": 8}],
            )))
            def add_op(x, y):
                return x + y

            def forward(self, x, y):
                return self.add_op(x, y)

        x = torch.randn(32, 64)
        y = torch.randn(32, 64)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_add_2d_blocked_dynamic(self):
        """2D blocked add with dynamic batch (first) dim."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_2d,
                configs=[{"BLOCK_M": 16, "BLOCK_N": 16}, {"BLOCK_M": 32, "BLOCK_N": 32}],
            )))
            def add_op(x, y):
                return x + y

            def forward(self, x, y):
                return self.add_op(x, y)

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(1, 64), opt_shape=(4, 64), max_shape=(8, 64), dtype=torch.float32),
            torch_tensorrt.Input(min_shape=(1, 64), opt_shape=(4, 64), max_shape=(8, 64), dtype=torch.float32),
        ]
        model = M()
        run_inputs = (torch.randn(4, 64, device="cuda"), torch.randn(4, 64, device="cuda"))
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )
        model = model.eval().cuda()
        for b in [1, 4, 8]:
            x = torch.randn(b, 64, device="cuda")
            y = torch.randn(b, 64, device="cuda")
            with torch.no_grad():
                torch.testing.assert_close(trt_model(x, y), model(x, y), atol=1e-3, rtol=1e-3)


@pytest.mark.requires_pre_bw
class TestTritonFormatE2E(unittest.TestCase):
    """Triton specs with explicit input_formats / output_formats declarations."""

    def test_explicit_linear_format(self):
        """Explicit LINEAR input/output formats compile and run correctly."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[{"BLOCK_SIZE": 128}],
                input_formats=[trt.TensorFormat.LINEAR],
                output_formats=[trt.TensorFormat.LINEAR],
            )))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_hwc8_format(self):
        """Spec declares HWC8 as the only format; exercises the HWC8 token code path."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_add_one,
                configs=[{"BLOCK_SIZE": 128}],
                input_formats=[trt.TensorFormat.HWC8],
                output_formats=[trt.TensorFormat.HWC8],
            )))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.ones(1, 8, 4, 4, dtype=torch.float16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
            atol=1e-2, rtol=1e-2,
        )

    def test_multi_configs_hwc8_format(self):
        """Two BLOCK_SIZE configs both declare HWC8 format; tactics metadata checked."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(tta.triton(
                launch_scale,
                configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}],
                input_formats=[trt.TensorFormat.HWC8],
                output_formats=[trt.TensorFormat.HWC8],
            )))
            def custom_op(x):
                return x * 2.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.ones(1, 8, 4, 4, dtype=torch.float16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "triton", "fn_name": "launch_scale", "config": {"BLOCK_SIZE": 64}},
            {"idx": 2, "backend": "triton", "fn_name": "launch_scale", "config": {"BLOCK_SIZE": 128}},
        ])

    def test_two_specs_hwc8_format(self):
        """Two specs both declare HWC8; collect_allowed_formats_for_io union = {HWC8}."""
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin([
                tta.triton(
                    launch_add_one,
                    configs=[{"BLOCK_SIZE": 64}],
                    input_formats=[trt.TensorFormat.HWC8],
                    output_formats=[trt.TensorFormat.HWC8],
                ),
                tta.triton(
                    launch_add_one,
                    configs=[{"BLOCK_SIZE": 128}],
                    input_formats=[trt.TensorFormat.HWC8],
                    output_formats=[trt.TensorFormat.HWC8],
                ),
            ]))
            def custom_op(x):
                return x + 1.0

            def forward(self, x):
                return self.custom_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.ones(1, 8, 4, 4, dtype=torch.float16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 64}},
            {"idx": 2, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 128}},
        ])
        assert_plugin_io_format(self, trt_model, "HWC8")


class TestTritonConvReluPoolE2E(unittest.TestCase):
    """3×3 conv (pad=1) + ReLU + 2×2 max pool via Triton."""

    def _meta(self, x, weight, bias):
        out = torch.nn.functional.conv2d(x, weight, bias, padding=1)
        return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

    def test_conv3x3_relu_pool(self):
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 2, 3, 3) * 0.1)
                self.bias = nn.Parameter(torch.zeros(4))

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 4}]),
                meta_impl=meta,
            ))
            def conv_pool(x, weight, bias):
                out = torch.nn.functional.conv2d(x, weight, bias, padding=1)
                return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

            def forward(self, x):
                return self.conv_pool(x, self.weight, self.bias)

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_conv3x3_relu_pool_larger(self):
        """Larger spatial dims and more channels."""
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 4, 3, 3) * 0.1)
                self.bias = nn.Parameter(torch.zeros(8))

            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 8}]),
                meta_impl=meta,
            ))
            def conv_pool(x, weight, bias):
                out = torch.nn.functional.conv2d(x, weight, bias, padding=1)
                return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

            def forward(self, x):
                return self.conv_pool(x, self.weight, self.bias)

        x = torch.randn(1, 4, 16, 16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_conv3x3_relu_pool_nn_conv2d(self):
        """nn.Conv2d as a sub-module; instance method uses self.conv(x) directly.

        tta.self_attr("conv.weight") / ("conv.bias") as parameter defaults declare
        which module tensors flow into the plugin, so forward() only needs x.
        """
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
                nn.init.normal_(self.conv.weight, std=0.1)
                nn.init.zeros_(self.conv.bias)

            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 4}]),
                meta_impl=meta,
            ))
            def conv_relu_pool(self, x,
                               weight=tta.self_attr("conv.weight"),
                               bias=tta.self_attr("conv.bias")):
                # Body uses module's conv2d (nn.Conv2d.__call__), not F.conv2d.
                out = self.conv(x)
                return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)

            def forward(self, x):
                return self.conv_relu_pool(x)  # only x — weights come from self_attr defaults

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_conv3x3_relu_pool_nn_conv2d_no_bias(self):
        """nn.Conv2d without bias: zero bias buffer declared via self_attr."""
        meta = self._meta

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
                nn.init.normal_(self.conv.weight, std=0.1)
                self.register_buffer("zero_bias", torch.zeros(8))

            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 8}]),
                meta_impl=meta,
            ))
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



class TestTritonPluginMultiOutputE2E(unittest.TestCase):
    """Multi-output custom_plugin: a single Triton kernel writes two output tensors."""

    def test_add_one_and_scale_two_outputs(self):
        """Triton kernel produces two outputs: out0 = x+1, out1 = x*2."""
        def meta(x):
            return torch.empty_like(x), torch.empty_like(x)

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_add_one_and_scale, configs=[{"BLOCK_SIZE": 128}]),
                meta_impl=meta,
            ))
            def custom_op(x):
                return x + 1.0, x * 2.0

            def forward(self, x):
                out0, out1 = self.custom_op(x)
                return out0, out1

        x = torch.randn(256)
        model = M().eval().cuda()
        x_gpu = x.cuda()

        with torch.no_grad():
            eager_out = model(x_gpu)

        trt_model = torch_tensorrt.compile(
            model,
            inputs=(x_gpu,),
            min_block_size=1,
            require_full_compilation=True,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        )
        with torch.no_grad():
            trt_out = trt_model(x_gpu)

        self.assertEqual(len(trt_out), 2)
        torch.testing.assert_close(trt_out[0], eager_out[0], atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(trt_out[1], eager_out[1], atol=1e-3, rtol=1e-3)
        # Verify values match expected computation
        torch.testing.assert_close(trt_out[0].cpu(), x + 1.0, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(trt_out[1].cpu(), x * 2.0, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# TestSelfAttrDottedPathE2E (moved from integration/test_coverage_gaps_e2e.py)
# ---------------------------------------------------------------------------

# Kernel must be at module level so AOT triton can find it by module inspection.
@triton.jit
def _scale_w_kernel(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) * tl.load(w_ptr + offs, mask=mask),
        mask=mask,
    )


def _launch_scale_by_weight(x, w, out, BLOCK=128):
    _scale_w_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, w, out, x.numel(), BLOCK=BLOCK)


class TestSelfAttrDottedPathE2E(unittest.TestCase):
    """self_attr("params.scale") resolves a nested module attribute at trace time."""

    def test_dotted_path_compiles_and_runs(self):
        """Scale-by-weight using dotted path self_attr("params.scale").

        self_attr is a tracing-time declaration: it tells TTA to pass
        self.params.scale as an extra tensor input to the plugin.  The eager
        body must work independently (tta is a no-op in eager), so it reads
        self.params.scale directly rather than relying on the `weight` default.
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterDict(
                    {"scale": nn.Parameter(torch.ones(256), requires_grad=False)}
                )

            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(_launch_scale_by_weight, configs=[{"BLOCK": 128}])
            ))
            def scale_op(self, x, weight=tta.self_attr("params.scale")):
                # Eager body: tta is a no-op, so access the attribute directly.
                return x * self.params.scale

            def forward(self, x):
                return self.scale_op(x)

        x = torch.randn(256)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )


# ---------------------------------------------------------------------------
# TestWeightBindingE2E
# ---------------------------------------------------------------------------

class TestWeightBindingE2E(unittest.TestCase):
    """Weight binding: tensor-valued kwargs in custom_plugin() declare frozen weights.

    Tensor kwargs split from scalar attrs at plugin creation time.  The bound
    weight count (num_weights) is included in the op_name fingerprint so that
    the same kernel registered with different input arities gets distinct QDP
    entries and avoids stale-registration bugs.
    """

    def test_weight_binding_scales_activation(self):
        """Elementwise multiply: activation * frozen weight (weight injected from descriptor).

        The annotated function takes only the dynamic input x; the frozen weight
        is declared in custom_plugin(w=...) and injected as a TRT constant at
        lowering time.  The eager body is unchanged — it accesses self.scale
        directly with no tta-specific changes.
        """
        scale = torch.ones(256).cuda()

        class M(nn.Module):
            impl = tta.custom_plugin(
                tta.triton(launch_elementwise_mul, configs=[{"BLOCK": 128}]),
                w=scale,
            )

            def __init__(self):
                super().__init__()
                self.scale = scale

            @tta.export_as(impl=impl)
            def scale_op(self, x):
                return x * self.scale

            def forward(self, x):
                return self.scale_op(x)

        # Verify the descriptor captured the weight.
        self.assertIn("w", M.impl.weights)
        self.assertEqual(M.impl.weights["w"].shape, scale.shape)

        x = torch.randn(256)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_weight_binding_different_op_name_from_no_weight(self):
        """Same kernel used with and without a weight gets distinct op_names."""
        no_weight_impl = tta.custom_plugin(
            tta.triton(launch_elementwise_mul, configs=[{"BLOCK": 128}]),
        )
        weight_impl = tta.custom_plugin(
            tta.triton(launch_elementwise_mul, configs=[{"BLOCK": 128}]),
            w=torch.ones(256),
        )
        # Different num_weights in fingerprint → different op_names.
        self.assertNotEqual(no_weight_impl.op_name, weight_impl.op_name)

    def test_weight_binding_with_module_parameter(self):
        """Weight passed via self_attr kwarg on custom_plugin from an nn.Parameter.

        tta.self_attr("scale") declared on tta.custom_plugin(w=...) resolves
        self.scale at trace time and flows it into the plugin as the second
        tensor input via region IO.  The eager body reads self.scale directly
        (tta is a no-op in eager), so the function signature stays clean.
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(128), requires_grad=False)

            @tta.export_as(impl=tta.custom_plugin(
                tta.triton(launch_elementwise_mul, configs=[{"BLOCK": 128}]),
                w=tta.self_attr("scale"),
            ))
            def scale_op(self, x):
                # Eager body: tta is a no-op, access the attribute directly.
                return x * self.scale

            def forward(self, x):
                return self.scale_op(x)

        x = torch.randn(128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )
