"""Unit tests for builtin annotation: eager accuracy, export graph, validation (CPU-only)."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation._registry import _spec_registry as _SPEC_REGISTRY
from torch_tensorrt.annotation._specs import BuiltinSpec
from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_ATEN_CONVERTERS


# ===================================================================
# 1. Basic eager + export
# ===================================================================


class TestBuiltinEagerAndExport(unittest.TestCase):
    """Builtin annotation: eager accuracy and export graph structure."""

    def test_simple_builtin_export(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def custom_relu(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        class SimpleModel(nn.Module):
            def forward(self, x):
                return custom_relu(x)

        model = SimpleModel()
        x = torch.randn(10, 20)
        eager_out = model(x)
        torch.testing.assert_close(eager_out, torch.relu(x), atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        self.assertIn("torch_tensorrt_anno_builtin", str(exported.graph))

    def test_builtin_with_params(self):
        @tta.export_as(impl=tta.builtin("add_constant", shape=(1,), weights=None))
        def add_five(x: torch.Tensor) -> torch.Tensor:
            return x + 5.0

        model = nn.Module()
        model.forward = lambda x: add_five(x)
        x = torch.randn(8, 8)
        torch.testing.assert_close(model(x), x + 5.0, atol=1e-5, rtol=1e-5)

    def test_convolution_builtin(self):
        @tta.export_as(
            impl=tta.builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        )
        def custom_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, weight, padding=1)

        x = torch.randn(1, 32, 28, 28)
        weight = torch.randn(64, 32, 3, 3)
        self.assertEqual(custom_conv(x, weight).shape, (1, 64, 28, 28))

    def test_multiple_builtin_ops(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_constant", shape=(1,), weights=None))
        def add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

        class M(nn.Module):
            def forward(self, x):
                return add_one(relu(x))

        model = M()
        x = torch.randn(5, 10) - 0.5
        torch.testing.assert_close(model(x), torch.relu(x) + 1.0, atol=1e-5, rtol=1e-5)

    def test_builtin_zero_inputs(self):
        @tta.export_as(impl=tta.builtin("add_constant", shape=(5, 5), weights=None))
        def get_constant() -> torch.Tensor:
            return torch.ones(5, 5)

        class M(nn.Module):
            def forward(self, x):
                return x + get_constant()

        model = M()
        x = torch.randn(5, 5)
        torch.testing.assert_close(model(x), x + torch.ones(5, 5), atol=1e-5, rtol=1e-5)
        exported = torch.export.export(model, (x,))
        self.assertIn("torch_tensorrt_anno_builtin", str(exported.graph))


# ===================================================================
# 2. Numerical accuracy per TRT layer type (CPU eager)
# ===================================================================


class TestBuiltinConvolutionEager(unittest.TestCase):
    """Convolution builtin: numerical accuracy in eager mode."""

    def test_conv2d_known_weights(self):
        @tta.export_as(impl=tta.builtin("add_convolution_nd", num_output_maps=2, kernel_shape=(3, 3), kernel=None))
        def conv2d_3x3(x, weight):
            return F.conv2d(x, weight, padding=1)

        weight = torch.zeros(2, 1, 3, 3)
        weight[0, 0, 1, 1] = 1.0
        weight[1, 0, :, :] = 1.0 / 9.0
        x = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
        result = conv2d_3x3(x, weight)
        self.assertEqual(result.shape, (1, 2, 5, 5))
        self.assertAlmostEqual(result[0, 0, 2, 2].item(), 12.0, places=4)
        self.assertAlmostEqual(result[0, 1, 2, 2].item(), 12.0, places=4)

    def test_conv2d_with_stride(self):
        @tta.export_as(impl=tta.builtin("add_convolution_nd", num_output_maps=4, kernel_shape=(3, 3), kernel=None))
        def conv2d_stride2(x, weight):
            return F.conv2d(x, weight, stride=2, padding=1)

        x = torch.randn(1, 3, 8, 8)
        weight = torch.randn(4, 3, 3, 3)
        self.assertEqual(conv2d_stride2(x, weight).shape, (1, 4, 4, 4))

    def test_conv2d_multiple_channels(self):
        @tta.export_as(impl=tta.builtin("add_convolution_nd", num_output_maps=16, kernel_shape=(3, 3), kernel=None))
        def conv2d_multi(x, weight):
            return F.conv2d(x, weight, padding=1)

        x = torch.randn(2, 8, 16, 16)
        weight = torch.randn(16, 8, 3, 3)
        self.assertEqual(conv2d_multi(x, weight).shape, (2, 16, 16, 16))


class TestBuiltinPoolingEager(unittest.TestCase):
    """Pooling builtin: numerical accuracy in eager mode."""

    def test_maxpool2d_known_values(self):
        @tta.export_as(impl=tta.builtin("add_pooling_nd", type=0, window_size=(2, 2)))
        def maxpool2x2(x):
            return F.max_pool2d(x, kernel_size=2)

        x = torch.tensor([[[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        result = maxpool2x2(x)
        self.assertEqual(result.shape, (1, 1, 2, 2))
        self.assertEqual(result[0, 0, 0, 0].item(), 6.0)
        self.assertEqual(result[0, 0, 0, 1].item(), 8.0)
        self.assertEqual(result[0, 0, 1, 0].item(), 14.0)
        self.assertEqual(result[0, 0, 1, 1].item(), 16.0)

    def test_avgpool2d_known_values(self):
        @tta.export_as(impl=tta.builtin("add_pooling_nd", type=1, window_size=(2, 2)))
        def avgpool2x2(x):
            return F.avg_pool2d(x, kernel_size=2)

        x = torch.tensor([[[[1., 2.], [3., 4.]]]])
        self.assertAlmostEqual(avgpool2x2(x)[0, 0, 0, 0].item(), 2.5, places=5)

    def test_pooling_with_stride(self):
        @tta.export_as(impl=tta.builtin("add_pooling_nd", type=0, window_size=(3, 3)))
        def maxpool3x3(x):
            return F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        self.assertEqual(maxpool3x3(torch.randn(1, 4, 7, 7)).shape, (1, 4, 4, 4))


class TestBuiltinMatMulEager(unittest.TestCase):
    """MatMul builtin: numerical accuracy in eager mode."""

    def test_matmul_known_values(self):
        @tta.export_as(impl=tta.builtin("add_matrix_multiply", op0=0, op1=0))
        def matmul(x, y):
            return torch.matmul(x, y)

        x = torch.tensor([[1., 2.], [3., 4.]])
        y = torch.tensor([[2., 0.], [1., 2.]])
        result = matmul(x, y)
        self.assertAlmostEqual(result[0, 0].item(), 4.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 4.0, places=5)
        self.assertAlmostEqual(result[1, 0].item(), 10.0, places=5)
        self.assertAlmostEqual(result[1, 1].item(), 8.0, places=5)

    def test_batched_matmul(self):
        @tta.export_as(impl=tta.builtin("add_matrix_multiply", op0=0, op1=0))
        def matmul(x, y):
            return torch.matmul(x, y)

        self.assertEqual(matmul(torch.randn(4, 10, 20), torch.randn(4, 20, 30)).shape, (4, 10, 30))


class TestBuiltinScaleEager(unittest.TestCase):
    """Scale builtin: numerical accuracy in eager mode."""

    def test_uniform_scale(self):
        @tta.export_as(impl=tta.builtin("add_scale", mode=0, shift=None, scale=None, power=None))
        def scale_layer(x):
            return x * 2.5

        x = torch.tensor([[1., 2., 3.]])
        result = scale_layer(x)
        self.assertAlmostEqual(result[0, 0].item(), 2.5, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 5.0, places=5)
        self.assertAlmostEqual(result[0, 2].item(), 7.5, places=5)

    def test_scale_with_shift(self):
        @tta.export_as(impl=tta.builtin("add_scale", mode=0, shift=None, scale=None, power=None))
        def scale_shift(x):
            return x * 2.0 + 1.0

        x = torch.tensor([[0., 1., 2.]])
        result = scale_shift(x)
        self.assertAlmostEqual(result[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 3.0, places=5)
        self.assertAlmostEqual(result[0, 2].item(), 5.0, places=5)


class TestBuiltinReduceEager(unittest.TestCase):
    """Reduce builtin: numerical accuracy in eager mode."""

    def test_reduce_sum(self):
        @tta.export_as(impl=tta.builtin("add_reduce", op=0, axes=1, keep_dims=True))
        def reduce_sum(x):
            return torch.sum(x, dim=1, keepdim=True)

        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        result = reduce_sum(x)
        self.assertEqual(result.shape, (2, 1))
        self.assertAlmostEqual(result[0, 0].item(), 6.0, places=5)
        self.assertAlmostEqual(result[1, 0].item(), 15.0, places=5)

    def test_reduce_mean(self):
        @tta.export_as(impl=tta.builtin("add_reduce", op=0, axes=1, keep_dims=True))
        def reduce_mean(x):
            return torch.mean(x, dim=1, keepdim=True)

        x = torch.tensor([[2., 4., 6.], [1., 3., 5.]])
        result = reduce_mean(x)
        self.assertAlmostEqual(result[0, 0].item(), 4.0, places=5)
        self.assertAlmostEqual(result[1, 0].item(), 3.0, places=5)

    def test_reduce_max(self):
        @tta.export_as(impl=tta.builtin("add_reduce", op=0, axes=2, keep_dims=False))
        def reduce_max(x):
            return torch.max(x, dim=2)[0]

        x = torch.tensor([[[1., 5., 3.], [2., 4., 6.]]])
        result = reduce_max(x)
        self.assertEqual(result.shape, (1, 2))
        self.assertAlmostEqual(result[0, 0].item(), 5.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 6.0, places=5)


class TestBuiltinElementwiseEager(unittest.TestCase):
    """Elementwise builtin: numerical accuracy in eager mode."""

    def test_multiply(self):
        @tta.export_as(impl=tta.builtin("add_elementwise", op=1))
        def multiply(x, y):
            return x * y

        x, y = torch.tensor([[2., 3.]]), torch.tensor([[4., 5.]])
        result = multiply(x, y)
        self.assertAlmostEqual(result[0, 0].item(), 8.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 15.0, places=5)

    def test_subtract(self):
        @tta.export_as(impl=tta.builtin("add_elementwise", op=2))
        def subtract(x, y):
            return x - y

        result = subtract(torch.tensor([[10., 20.]]), torch.tensor([[3., 7.]]))
        self.assertAlmostEqual(result[0, 0].item(), 7.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 13.0, places=5)

    def test_divide(self):
        @tta.export_as(impl=tta.builtin("add_elementwise", op=3))
        def divide(x, y):
            return x / y

        result = divide(torch.tensor([[10., 20.]]), torch.tensor([[2., 4.]]))
        self.assertAlmostEqual(result[0, 0].item(), 5.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 5.0, places=5)


class TestBuiltinActivationEager(unittest.TestCase):
    """Activation builtin: numerical accuracy in eager mode."""

    def test_sigmoid(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=1))
        def sigmoid(x):
            return torch.sigmoid(x)

        result = sigmoid(torch.tensor([[0., 1., -1.]]))
        self.assertAlmostEqual(result[0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 0.731, places=2)
        self.assertAlmostEqual(result[0, 2].item(), 0.269, places=2)

    def test_tanh(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=2))
        def tanh(x):
            return torch.tanh(x)

        result = tanh(torch.tensor([[0., 1., -1.]]))
        self.assertAlmostEqual(result[0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 0.762, places=2)
        self.assertAlmostEqual(result[0, 2].item(), -0.762, places=2)

    def test_leaky_relu(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0, alpha=0.1))
        def leaky_relu(x):
            return F.leaky_relu(x, negative_slope=0.1)

        result = leaky_relu(torch.tensor([[-10., 0., 10.]]))
        self.assertAlmostEqual(result[0, 0].item(), -1.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(result[0, 2].item(), 10.0, places=5)


# ===================================================================
# 3. Complex patterns (CPU eager)
# ===================================================================


class TestBuiltinComplexPatternsEager(unittest.TestCase):
    """Complex real-world patterns using builtin annotations (CPU eager)."""

    def test_batch_norm_pattern(self):
        @tta.export_as(impl=tta.builtin("add_scale", mode=0, shift=None, scale=None, power=None))
        def scale_shift(x):
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True) + 1e-5
            return (x - mean) / std

        x = torch.tensor([[1., 2., 3., 4., 5.]])
        result = scale_shift(x)
        self.assertAlmostEqual(torch.mean(result).item(), 0.0, places=1)
        self.assertAlmostEqual(torch.std(result).item(), 1.0, places=1)

    def test_attention_qkv_pattern(self):
        @tta.export_as(impl=tta.builtin("add_matrix_multiply", op0=0, op1=1))
        def qk_product(q, k_t):
            return torch.matmul(q, k_t)

        self.assertEqual(qk_product(torch.randn(2, 4, 8), torch.randn(2, 8, 4)).shape, (2, 4, 4))

    def test_residual_with_projection(self):
        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def residual_add(x, y):
            return x + y

        class ResidualProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(10, 10)

            def forward(self, x):
                return residual_add(self.proj(x), x)

        model = ResidualProjection().eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            self.assertEqual(model(x).shape, (4, 10))

    def test_resnet_block_pattern(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def add(x, y):
            return x + y

        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
                self.conv2 = nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, x):
                out = relu(self.conv2(relu(self.conv1(x))))
                return relu(add(out, x))

        model = ResidualBlock().eval()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            self.assertEqual(model(x).shape, (1, 3, 8, 8))

        exported = torch.export.export(model, (x,))
        self.assertIn("torch_tensorrt_anno_builtin", str(exported.graph))

    def test_chain_of_operations(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def add(x, y):
            return x + y

        class M(nn.Module):
            def forward(self, x):
                x = relu(x)
                x = add(x, torch.ones_like(x))
                return relu(x)

        model = M().eval()
        x = torch.tensor([[-5., -1., 0., 1., 5.]])
        with torch.no_grad():
            result = model(x)
        expected = torch.tensor([[1., 1., 1., 2., 6.]])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_linear_plus_relu(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 3)
                with torch.no_grad():
                    self.linear.weight.copy_(torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.]]))
                    self.linear.bias.copy_(torch.tensor([0., 0., 0.]))

            def forward(self, x):
                return relu(self.linear(x))

        model = M().eval()
        x = torch.tensor([[-1., 2., -3., 4., 5.]])
        with torch.no_grad():
            result = model(x)
        torch.testing.assert_close(result, torch.tensor([[0., 2., 0.]]), atol=1e-5, rtol=1e-5)


# ===================================================================
# 4. Registry (CPU, export-only)
# ===================================================================


class TestBuiltinRegistry(unittest.TestCase):
    """Op registry and converter side-table for builtin annotations."""

    def test_same_spec_reuses_op(self):
        import tensorrt as trt

        class MA(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def forward(self, x):
                return self.relu(x)

        class MB(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def forward(self, x):
                return self.relu(x)

        x = torch.randn(2, 4)
        ea = torch.export.export(MA(), (x,))
        eb = torch.export.export(MB(), (x,))

        targets_a = {str(n.target) for n in ea.graph.nodes if "torch_tensorrt_anno_builtin" in str(n.target)}
        targets_b = {str(n.target) for n in eb.graph.nodes if "torch_tensorrt_anno_builtin" in str(n.target)}
        self.assertEqual(targets_a, targets_b)

    def test_different_specs_different_ops(self):
        import tensorrt as trt

        class MR(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def act(x):
                return torch.relu(x)

            def forward(self, x):
                return self.act(x)

        class MS(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID))
            def act(x):
                return torch.sigmoid(x)

            def forward(self, x):
                return self.act(x)

        x = torch.randn(2, 4)
        er = torch.export.export(MR(), (x,))
        es = torch.export.export(MS(), (x,))

        tr = {str(n.target) for n in er.graph.nodes if "torch_tensorrt_anno_builtin" in str(n.target)}
        ts = {str(n.target) for n in es.graph.nodes if "torch_tensorrt_anno_builtin" in str(n.target)}
        self.assertNotEqual(tr, ts)

    def test_spec_in_side_table(self):
        import tensorrt as trt

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.TANH))
            def tanh(x):
                return torch.tanh(x)

            def forward(self, x):
                return self.tanh(x)

        torch.export.export(M(), (torch.randn(2, 4),))
        found = any(
            isinstance(spec, BuiltinSpec) and spec.add_name == "add_activation"
            and spec.kwargs.get("type") == trt.ActivationType.TANH
            for spec in _SPEC_REGISTRY.values()
        )
        self.assertTrue(found)

    def test_converter_registered_for_each_op(self):
        import tensorrt as trt

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUM))
            def add(a, b):
                return a + b

            def forward(self, a, b):
                return self.add(a, b)

        torch.export.export(M(), (torch.randn(2, 4), torch.randn(2, 4)))
        found = any("torch_tensorrt_anno_builtin" in str(op) for op in DYNAMO_ATEN_CONVERTERS)
        self.assertTrue(found)


# ===================================================================
# 5. Error handling (CPU eager)
# ===================================================================


class TestBuiltinEagerErrors(unittest.TestCase):
    """Error conditions in eager mode."""

    def test_shape_mismatch(self):
        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def add(x, y):
            return x + y

        with self.assertRaises(RuntimeError):
            add(torch.randn(3, 5), torch.randn(3, 4))

    def test_dtype_preserved(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(4, 4, dtype=dtype)
            self.assertEqual(relu(x).dtype, dtype)

    def test_nan_propagation(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        result = relu(torch.tensor([[1., float("nan"), 3.]]))
        self.assertTrue(torch.isnan(result[0, 1]))
        self.assertEqual(result[0, 0].item(), 1.0)

    def test_dynamic_batch_size(self):
        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        model = nn.Module()
        model.forward = lambda x: relu(x)
        model.eval()
        for bs in [1, 4, 8, 16]:
            with torch.no_grad():
                result = model(torch.randn(bs, 10))
            self.assertEqual(result.shape, (bs, 10))
            self.assertTrue(torch.all(result >= 0))
