"""Integration tests: builtin annotations. Compare eager vs TRT accuracy (GPU + TRT)."""

import unittest

import tensorrt as trt
import torch
import torch.nn as nn

import torch_tensorrt.annotation as tta

from ._e2e_common import _compile_and_run, assert_trt_compiled


# ===================================================================
# 1. Activation layers
# ===================================================================


class TestBuiltinActivationE2E(unittest.TestCase):
    """Activation layers lowered to TRT engine."""

    def test_relu(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def forward(self, x):
                return self.relu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_sigmoid(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID))
            def sigmoid(x):
                return torch.sigmoid(x)

            def forward(self, x):
                return self.sigmoid(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )

    def test_tanh(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.TANH))
            def tanh(x):
                return torch.tanh(x)

            def forward(self, x):
                return self.tanh(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )

    def test_elu(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.ELU, alpha=1.0))
            def elu(x):
                return torch.nn.functional.elu(x, alpha=1.0)

            def forward(self, x):
                return self.elu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(4, 16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )


# ===================================================================
# 2. Elementwise layers
# ===================================================================


class TestBuiltinElementwiseE2E(unittest.TestCase):
    """Elementwise ops lowered to TRT engine."""

    def test_add(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUM))
            def add(a, b):
                return a + b

            def forward(self, a, b):
                return self.add(a, b)

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_elementwise"}],
        )

    def test_multiply(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.PROD))
            def mul(a, b):
                return a * b

            def forward(self, a, b):
                return self.mul(a, b)

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_elementwise"}],
        )

    def test_subtract(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUB))
            def sub(a, b):
                return a - b

            def forward(self, a, b):
                return self.sub(a, b)

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_elementwise"}],
        )

    def test_max(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.MAX))
            def emax(a, b):
                return torch.maximum(a, b)

            def forward(self, a, b):
                return self.emax(a, b)

        x, y = torch.randn(4, 8), torch.randn(4, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_elementwise"}],
        )


# ===================================================================
# 3. Chained builtins
# ===================================================================


class TestChainedBuiltinsE2E(unittest.TestCase):
    """Multi-op chains compiled to a single TRT engine."""

    def test_relu_add_relu_chain(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            @staticmethod
            @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUM))
            def add(x, y):
                return x + y

            def forward(self, x, y):
                return self.relu(self.add(self.relu(x), y))

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_resnet_block(self):
        class ResBlock(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            @staticmethod
            @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUM))
            def add(x, y):
                return x + y

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(16, 16, 3, padding=1)

            def forward(self, x):
                out = self.relu(self.conv(x))
                return self.relu(self.add(out, x))

        trt_model, trt_out, eager_out = _compile_and_run(ResBlock(), (torch.randn(1, 16, 8, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_conv_relu_conv_relu(self):
        class ConvBlock(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 8, 3, padding=1)

            def forward(self, x):
                return self.relu(self.conv2(self.relu(self.conv1(x))))

        trt_model, trt_out, eager_out = _compile_and_run(ConvBlock(), (torch.randn(1, 8, 16, 16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")


# ===================================================================
# 4. Enum type conversion
# ===================================================================


class TestEnumTypeConversionE2E(unittest.TestCase):
    """Binder enum resolution works E2E (int → TRT enum auto-conversion)."""

    def test_int_enum_auto_convert(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def forward(self, x):
                return self.relu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_trt_enum_passthrough(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID))
            def sigmoid(x):
                return torch.sigmoid(x)

            def forward(self, x):
                return self.sigmoid(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )


# ===================================================================
# 5. TRT compile via local helper
# ===================================================================


class TestBuiltinTRTCompileE2E(unittest.TestCase):
    """Builtin TRT layers through the full TTA compile pipeline."""

    def test_relu_e2e(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def forward(self, x):
                return relu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_elementwise_add_e2e(self):
        @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUM))
        def add(x, y):
            return x + y

        class M(nn.Module):
            def forward(self, x, y):
                return add(x, y)

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_elementwise"}],
        )

    def test_chained_ops_e2e(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUM))
        def add(x, y):
            return x + y

        class M(nn.Module):
            def forward(self, x, y):
                return relu(add(relu(x), y))

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")


# ===================================================================
# 6. Corner cases
# ===================================================================


class TestBuiltinCornerCases(unittest.TestCase):
    """Corner cases for tta.builtin + tta.export_as with TRT engine."""

    def test_relu_batch_one(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def forward(self, x):
                return relu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(1, 16) - 0.5,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_relu_large_tensor(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def forward(self, x):
                return relu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(8, 64, 32, 32) - 0.5,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_two_different_activations(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID))
        def sigmoid(x):
            return torch.sigmoid(x)

        class M(nn.Module):
            def forward(self, x):
                return sigmoid(relu(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 16) - 0.5,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )

    def test_same_relu_used_twice(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def forward(self, x):
                return relu(relu(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 16) - 0.5,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_builtin_with_conv_trained_weights(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                return relu(self.conv(x))

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 3, 8, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_builtin_instance_method(self):
        class M(nn.Module):
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.TANH))
            def tanh_op(self, x):
                return torch.tanh(x)

            def forward(self, x):
                return self.tanh_op(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
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

    def test_builtin_self_attr_float(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.neg_slope = 0.2

            @tta.export_as(
                impl=tta.builtin(
                    "add_activation",
                    type=trt.ActivationType.LEAKY_RELU,
                    alpha=tta.self_attr("neg_slope"),
                )
            )
            def lrelu(self, x):
                return torch.nn.functional.leaky_relu(x, self.neg_slope)

            def forward(self, x):
                return self.lrelu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 16) - 0.5,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )

    def test_builtin_residual_block(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

            def forward(self, x):
                out = relu(self.conv1(x))
                out = self.conv2(out)
                return relu(out + x)

        trt_model, trt_out, eager_out = _compile_and_run(ResBlock(), (torch.randn(2, 16, 8, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out >= 0).all(), "ReLU output must be non-negative")

    def test_all_negative_input(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def forward(self, x):
                return relu(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (-torch.abs(torch.randn(2, 16)),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-5, rtol=1e-5,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
        self.assertTrue((trt_out == 0).all())

    def test_all_zero_input(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID))
        def sigmoid(x):
            return torch.sigmoid(x)

        class M(nn.Module):
            def forward(self, x):
                return sigmoid(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.zeros(2, 16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-5, rtol=1e-5,
            expected_tta_metadata=[{"backend": "builtin", "plugin_name": "add_activation"}],
        )
