"""Unit tests for @export_as with custom_plugin (Triton, CuTile, CuTeDSL). CPU-only, stub kernels."""

import unittest

import torch
import torch.nn as nn

import torch_tensorrt.annotation as tta


class TestExportAsCustomPluginEager(unittest.TestCase):
    """export_as with custom_plugin specs works in eager mode for all backends."""

    def test_triton_eager(self):
        def triton_kernel(x, out):
            pass

        @tta.export_as(impl=tta.custom_plugin(tta.triton(triton_kernel)))
        def scale(x):
            return x * 2.0

        x = torch.randn(4, 4)
        torch.testing.assert_close(scale(x), x * 2.0, atol=1e-5, rtol=1e-5)

    def test_triton_with_configs_eager(self):
        def block_kernel(x, out, BLOCK: int):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(
                tta.triton(block_kernel, configs=[{"BLOCK": 64}, {"BLOCK": 128}])
            )
        )
        def relu_custom(x):
            return torch.relu(x)

        x = torch.randn(8, 8)
        torch.testing.assert_close(relu_custom(x), torch.relu(x), atol=1e-5, rtol=1e-5)

    def test_triton_with_meta_impl_eager(self):
        def triton_kernel(x, out):
            pass

        def meta_fn(x):
            return torch.empty_like(x)

        @tta.export_as(
            impl=tta.custom_plugin(tta.triton(triton_kernel), meta_impl=meta_fn)
        )
        def add_one(x):
            return x + 1

        x = torch.randn(3, 3)
        torch.testing.assert_close(add_one(x), x + 1, atol=1e-5, rtol=1e-5)

    def test_triton_multiple_tensor_inputs_eager(self):
        def binary_kernel(x, y, out):
            pass

        @tta.export_as(impl=tta.custom_plugin(tta.triton(binary_kernel)))
        def custom_add(x, y):
            return x + y

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        torch.testing.assert_close(custom_add(x, y), x + y, atol=1e-5, rtol=1e-5)

    def test_cutile_eager(self):
        def cutile_prog(x, out):
            pass

        @tta.export_as(impl=tta.custom_plugin(tta.cutile(cutile_prog)))
        def scale(x):
            return x * 3.0

        x = torch.randn(4, 4)
        torch.testing.assert_close(scale(x), x * 3.0, atol=1e-5, rtol=1e-5)

    def test_cutile_with_configs_eager(self):
        def tile_prog(x, out, TILE_M: int):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(
                tta.cutile(tile_prog, configs=[{"TILE_M": 32}, {"TILE_M": 64}])
            )
        )
        def custom_op(x):
            return torch.relu(x)

        x = torch.randn(8, 8)
        torch.testing.assert_close(custom_op(x), torch.relu(x), atol=1e-5, rtol=1e-5)

    def test_cutedsl_eager(self):
        def cute_kernel(x, out):
            pass

        @tta.export_as(impl=tta.custom_plugin(tta.cutedsl(cute_kernel, arch="sm_80")))
        def matmul_custom(a, b):
            return a @ b

        a = torch.randn(4, 8)
        b = torch.randn(8, 4)
        torch.testing.assert_close(matmul_custom(a, b), a @ b, atol=1e-5, rtol=1e-5)

    def test_cutedsl_no_arch_eager(self):
        def kernel(x, out):
            pass

        @tta.export_as(impl=tta.custom_plugin(tta.cutedsl(kernel)))
        def custom_sigmoid(x):
            return torch.sigmoid(x)

        x = torch.randn(4, 4)
        torch.testing.assert_close(custom_sigmoid(x), torch.sigmoid(x), atol=1e-5, rtol=1e-5)

    def test_cutedsl_with_configs_eager(self):
        def fused_attn(q, k, v, out, BLOCK: int):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(
                tta.cutedsl(fused_attn, arch="sm_90", configs=[{"BLOCK": 128}])
            )
        )
        def attention_op(q, k, v):
            return q + k + v

        q = torch.randn(4, 4)
        k = torch.randn(4, 4)
        v = torch.randn(4, 4)
        torch.testing.assert_close(attention_op(q, k, v), q + k + v, atol=1e-5, rtol=1e-5)

    def test_multi_backend_eager(self):
        def k1(x, out):
            pass

        def k2(x, out):
            pass

        @tta.export_as(impl=tta.custom_plugin([tta.triton(k1), tta.cutile(k2)]))
        def multi_op(x):
            return x * x

        x = torch.randn(4, 4)
        torch.testing.assert_close(multi_op(x), x * x, atol=1e-5, rtol=1e-5)

    def test_all_backends_produce_same_eager_result(self):
        def fn(x, out):
            pass

        @tta.export_as(impl=tta.custom_plugin(tta.triton(fn)))
        def triton_op(x):
            return x + 1.0

        @tta.export_as(impl=tta.custom_plugin(tta.cutile(fn)))
        def cutile_op(x):
            return x + 1.0

        @tta.export_as(impl=tta.custom_plugin(tta.cutedsl(fn, arch="sm_80")))
        def cutedsl_op(x):
            return x + 1.0

        x = torch.randn(4, 4)
        expected = x + 1.0
        torch.testing.assert_close(triton_op(x), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cutile_op(x), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cutedsl_op(x), expected, atol=1e-5, rtol=1e-5)

    def test_normalize_impl_passes_through_descriptor(self):
        from torch_tensorrt.annotation._specs import normalize_impl_to_spec

        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel))
        normalized = normalize_impl_to_spec(descriptor)
        self.assertIs(normalized, descriptor)

    def test_cutile_scale_3_5_python_body_accuracy(self):
        def cutile_scale_program(x_ptr, out_ptr, n_elements):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(tta.cutile(cutile_scale_program, configs=[{"TILE_SIZE": 128}]))
        )
        def scale_vector(x):
            return x * 3.5

        x = torch.randn(256, dtype=torch.float32)
        torch.testing.assert_close(scale_vector(x), x * 3.5)

    def test_cutile_relu_python_body_accuracy(self):
        def cutile_relu_program(x_ptr, out_ptr, n):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(
                tta.cutile(cutile_relu_program, configs=[{"TILE_M": 32}, {"TILE_M": 64}])
            )
        )
        def relu_op(x):
            return torch.relu(x)

        x = torch.randn(128, dtype=torch.float32)
        torch.testing.assert_close(relu_op(x), torch.relu(x))

    def test_cutedsl_elementwise_mul_python_body_accuracy(self):
        def cutedsl_mul_kernel(x_ptr, y_ptr, out_ptr, n):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(tta.cutedsl(cutedsl_mul_kernel, arch="sm_80"))
        )
        def elementwise_mul(x, y):
            return x * y

        x = torch.randn(512, dtype=torch.float32)
        y = torch.randn(512, dtype=torch.float32)
        torch.testing.assert_close(elementwise_mul(x, y), x * y)

    def test_cutedsl_sigmoid_python_body_accuracy(self):
        def cutedsl_sigmoid_kernel(x_ptr, out_ptr, n, BLOCK: int):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(
                tta.cutedsl(cutedsl_sigmoid_kernel, arch="sm_80", configs=[{"BLOCK": 64}, {"BLOCK": 128}])
            )
        )
        def sigmoid_op(x):
            return torch.sigmoid(x)

        x = torch.randn(4, 64, dtype=torch.float32)
        torch.testing.assert_close(sigmoid_op(x), torch.sigmoid(x))


class TestExportAsCustomPluginGraph(unittest.TestCase):
    """Custom plugin export_as produces correct graph ops (no GPU)."""

    def test_triton_graph_export(self):
        def triton_kernel(x_ptr, out_ptr, n, BLOCK: int):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(tta.triton(triton_kernel, configs=[{"BLOCK": 128}]))
        )
        def vector_scale(x):
            return x * 3.0

        class M(nn.Module):
            def forward(self, x):
                return vector_scale(x)

        exported = torch.export.export(M().eval(), (torch.randn(16),))
        self.assertIn("torch_tensorrt_anno_custom_plugin", str(exported.graph))

    def test_cutile_graph_export(self):
        def tile_prog(x_ptr, out_ptr, n):
            pass

        @tta.export_as(impl=tta.custom_plugin(tta.cutile(tile_prog)))
        def tile_scale(x):
            return x + 1.0

        class M(nn.Module):
            def forward(self, x):
                return tile_scale(x)

        exported = torch.export.export(M().eval(), (torch.randn(16),))
        self.assertIn("torch_tensorrt_anno_custom_plugin", str(exported.graph))

    def test_cutedsl_graph_export(self):
        def cute_kernel(x_ptr, out_ptr, n, BLOCK: int):
            pass

        @tta.export_as(
            impl=tta.custom_plugin(
                tta.cutedsl(cute_kernel, arch="sm_80", configs=[{"BLOCK": 64}])
            )
        )
        def custom_relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def forward(self, x):
                return custom_relu(x)

        exported = torch.export.export(M().eval(), (torch.randn(16),))
        self.assertIn("torch_tensorrt_anno_custom_plugin", str(exported.graph))

    def test_multi_backend_graph_export(self):
        def triton_impl(x_ptr, out_ptr, n, BLOCK: int):
            pass

        def cutile_impl(x_ptr, out_ptr, n):
            pass

        def cutedsl_impl(x_ptr, out_ptr, n):
            pass

        multi_spec = tta.custom_plugin([
            tta.triton(triton_impl, configs=[{"BLOCK": 128}]),
            tta.cutile(cutile_impl),
            tta.cutedsl(cutedsl_impl, arch="sm_80"),
        ])

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=multi_spec)
            def relu_custom(x):
                return torch.relu(x)

            def forward(self, x):
                return self.relu_custom(x)

        exported = torch.export.export(M().eval(), (torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]]),))
        self.assertIn("torch_tensorrt_anno_custom_plugin", str(exported.graph))

    def test_custom_plugin_with_nn_module_graph(self):
        def triton_act(x_ptr, out_ptr, n, BLOCK: int):
            pass

        class ConvBlock(nn.Module):
            @staticmethod
            @tta.export_as(
                impl=tta.custom_plugin(tta.triton(triton_act, configs=[{"BLOCK": 128}]))
            )
            def custom_gelu(x):
                return torch.nn.functional.gelu(x)

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                x = self.conv(x)
                x = self.custom_gelu(x)
                return x

        exported = torch.export.export(ConvBlock().eval(), (torch.randn(1, 3, 32, 32),))
        self.assertIn("torch_tensorrt_anno_custom_plugin", str(exported.graph))

    def test_triton_two_ops_in_graph(self):
        def triton_kernel(x_ptr, out_ptr, n, BLOCK: int):
            pass

        class M(nn.Module):
            @staticmethod
            @tta.export_as(
                impl=tta.custom_plugin(tta.triton(triton_kernel, configs=[{"BLOCK": 128}]))
            )
            def vector_add(x, y):
                return x + y

            def forward(self, x, y):
                return self.vector_add(x, y)

        exported = torch.export.export(M().eval(), (torch.randn(128), torch.randn(128)))
        self.assertIn("torch_tensorrt_anno_custom_plugin", str(exported.graph))
