"""
E2E tests for tta.plugin (InstanceNormalization_TRT v3): compare eager vs TRT accuracy.

Requires cuDNN 8; _ensure_cudnn8() auto-discovers libcudnn.so.8 from pip-installed nvidia-cudnn-cu12.
"""

import ctypes
import os
import sys
import unittest

import numpy as np


def _ensure_cudnn8():
    try:
        ctypes.CDLL("libcudnn.so.8")
        return True
    except OSError:
        pass
    for p in sys.path:
        if "site-packages" not in p:
            continue
        for root, _, files in os.walk(p):
            if any(f.startswith("libcudnn.so.8") for f in files):
                os.environ["LD_LIBRARY_PATH"] = root + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                try:
                    ctypes.CDLL(os.path.join(root, "libcudnn.so.8"))
                    return True
                except OSError:
                    pass
    return False


import tensorrt as trt
import torch
import torch.nn as nn

import torch_tensorrt
import torch_tensorrt.annotation as tta

from ._e2e_common import _compile_and_run, assert_trt_compiled

trt.init_libnvinfer_plugins(None, "")

C = 8
INSTNORM_KWARGS = dict(
    epsilon=1e-5,
    scales=np.ones(C, dtype=np.float32),
    bias=np.zeros(C, dtype=np.float32),
)


class TestPluginE2E(unittest.TestCase):
    """Export -> TRT compile -> run; compare TRT output to eager."""

    @classmethod
    def setUpClass(cls):
        if not _ensure_cudnn8():
            raise RuntimeError("libcudnn.so.8 not found; cannot run cuDNN plugin tests")

    def test_single_plugin(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def forward(self, x):
                return instnorm(x)

        x = torch.randn(2, C, 4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_two_chained_plugins(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def norm_a(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def norm_b(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def forward(self, x):
                return norm_b(norm_a(x))

        x = torch.randn(2, C, 4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_plugin_then_builtin(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        class M(nn.Module):
            def forward(self, x):
                return relu(instnorm(x))

        x = torch.randn(2, C, 4, 4) - 0.5
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_builtin_then_plugin(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def forward(self, x):
                return instnorm(relu(x))

        x = torch.randn(2, C, 4, 4) - 0.5
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_self_attr_scales_bias(self):
        eps = 1e-5

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(C))
                self.bias_param = nn.Parameter(torch.zeros(C))

            @tta.export_as(
                impl=tta.plugin(
                    "InstanceNormalization_TRT", "3", "",
                    epsilon=eps,
                    scales=tta.self_attr("weight"),
                    bias=tta.self_attr("bias_param"),
                )
            )
            def instnorm(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.instance_norm(x, self.weight, self.bias_param, eps=eps)

            def forward(self, x):
                return self.instnorm(x)

        x = torch.randn(2, C, 4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_builtin_self_attr(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 1.0

            @tta.export_as(
                impl=tta.builtin(
                    "add_activation",
                    type=trt.ActivationType.ELU,
                    alpha=tta.self_attr("alpha"),
                )
            )
            def elu_op(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.elu(x, alpha=self.alpha)

            def forward(self, x):
                return self.elu_op(x)

        x = torch.randn(2, 8) - 0.5
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )


class TestPluginCornerCases(unittest.TestCase):
    """Corner cases for tta.plugin + tta.export_as."""

    @classmethod
    def setUpClass(cls):
        if not _ensure_cudnn8():
            raise RuntimeError("libcudnn.so.8 not found; cannot run cuDNN plugin tests")

    def test_batch_size_one(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def forward(self, x):
                return instnorm(x)

        x = torch.randn(1, C, 4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_large_spatial(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def forward(self, x):
                return instnorm(x)

        x = torch.randn(2, C, 64, 64)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_small_spatial(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def forward(self, x):
                return instnorm(x)

        x = torch.randn(2, C, 2, 2)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_nonsquare_spatial(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def forward(self, x):
                return instnorm(x)

        x = torch.randn(2, C, 16, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_plugin_reused_across_instances(self):
        spec = tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS)

        @tta.export_as(impl=spec)
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M1(nn.Module):
            def forward(self, x):
                return instnorm(x)

        class M2(nn.Module):
            def forward(self, x):
                return instnorm(x) + 1.0

        x = torch.randn(2, C, 4, 4)
        trt_model1, trt_out1, eager_out1 = _compile_and_run(M1(), (x,))
        assert_trt_compiled(
            self, trt_model1, trt_out1, eager_out1,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )
        trt_model2, trt_out2, eager_out2 = _compile_and_run(M2(), (x,))
        assert_trt_compiled(
            self, trt_model2, trt_out2, eager_out2,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_plugin_after_conv(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(C, C, 3, padding=1)

            def forward(self, x):
                return instnorm(self.conv(x))

        x = torch.randn(2, C, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_plugin_between_linears(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(C, C, 1)
                self.conv2 = nn.Conv2d(C, C, 1)

            def forward(self, x):
                return self.conv2(instnorm(self.conv1(x)))

        x = torch.randn(2, C, 4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_self_attr_with_conv(self):
        eps = 1e-5

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(C, C, 3, padding=1)
                self.weight = nn.Parameter(torch.ones(C))
                self.bias_param = nn.Parameter(torch.zeros(C))

            @tta.export_as(
                impl=tta.plugin(
                    "InstanceNormalization_TRT", "3", "",
                    epsilon=eps,
                    scales=tta.self_attr("weight"),
                    bias=tta.self_attr("bias_param"),
                )
            )
            def instnorm(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.instance_norm(x, self.weight, self.bias_param, eps=eps)

            def forward(self, x):
                return self.instnorm(self.conv(x))

        x = torch.randn(2, C, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_same_plugin_different_eps(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "",
                                        epsilon=1e-3, scales=np.ones(C, dtype=np.float32),
                                        bias=np.zeros(C, dtype=np.float32)))
        def norm_coarse(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-3)

        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "",
                                        epsilon=1e-6, scales=np.ones(C, dtype=np.float32),
                                        bias=np.zeros(C, dtype=np.float32)))
        def norm_fine(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-6)

        class M(nn.Module):
            def forward(self, x):
                return norm_fine(norm_coarse(x))

        x = torch.randn(2, C, 4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )


class TestMixedPluginBuiltinCornerCases(unittest.TestCase):
    """Corner cases mixing tta.plugin and tta.builtin in the same model."""

    @classmethod
    def setUpClass(cls):
        if not _ensure_cudnn8():
            raise RuntimeError("libcudnn.so.8 not found; cannot run cuDNN plugin tests")

    def test_plugin_and_builtin_in_residual(self):
        @tta.export_as(impl=tta.plugin("InstanceNormalization_TRT", "3", "", **INSTNORM_KWARGS))
        def instnorm(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(x, eps=1e-5)

        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(C, C, 3, padding=1)

            def forward(self, x):
                out = relu(instnorm(self.conv(x)))
                return out + x

        x = torch.randn(2, C, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[
                {"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"},
                {"backend": "builtin", "plugin_name": "add_activation"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
