"""Unit tests for @export_as with plugin spec: graph structure, attrs, converter registration. CPU-only."""

import unittest

import torch
import torch.nn as nn

import torch_tensorrt.annotation as tta
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_ATEN_CONVERTERS,
)


class TestExportAsPluginGraph(unittest.TestCase):
    """Plugin ops appear in exported graph (CPU-only, no TRT runtime)."""

    def test_plugin_in_export_graph(self):
        """Plugin op node in FX graph after export."""

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.plugin("TestPlugin", "1.0", "test_ns"))
            def custom_op(x):
                return x * 2

            def forward(self, x):
                return self.custom_op(x)

        x = torch.randn(10, 20)
        exported = torch.export.export(M(), (x,))
        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_plugin", graph_str)

    def test_plugin_with_attrs_in_graph(self):
        """Plugin attrs (int, float, str, list) are captured in spec."""

        spec = tta.plugin(
            "AttrPlugin", "1.0", "test_ns",
            alpha=0.5, beta=2, mode="linear", dimensions=[1, 2, 3],
        )

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=spec)
            def custom_op(x):
                return x * 0.5 * 2.0

            def forward(self, x):
                return self.custom_op(x)

        x = torch.randn(4, 8)
        exported = torch.export.export(M(), (x,))
        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_plugin", graph_str)

        self.assertEqual(spec.attrs["alpha"], 0.5)
        self.assertEqual(spec.attrs["beta"], 2)
        self.assertEqual(spec.attrs["mode"], "linear")
        self.assertEqual(spec.attrs["dimensions"], [1, 2, 3])

    def test_multiple_plugins_in_graph(self):
        """Two different plugins in same model produce distinct graph ops."""

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.plugin("OpA", "1.0", "test_ns"))
            def op_a(x):
                return x * 2

            @staticmethod
            @tta.export_as(impl=tta.plugin("OpB", "1.0", "test_ns"))
            def op_b(x):
                return x + 1

            def forward(self, x):
                return self.op_b(self.op_a(x))

        x = torch.randn(5, 10)
        exported = torch.export.export(M(), (x,))
        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_plugin", graph_str)

    def test_plugin_converter_registered(self):
        """After export, plugin op has a converter in DYNAMO_ATEN_CONVERTERS."""

        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.plugin("RegTestPlugin", "1.0", "test_ns"))
            def registered_op(x):
                return x + 1

            def forward(self, x):
                return self.registered_op(x)

        x = torch.randn(4, 4)
        torch.export.export(M(), (x,))

        found = any(
            "torch_tensorrt_anno_plugin" in str(op)
            for op in DYNAMO_ATEN_CONVERTERS
        )
        self.assertTrue(found, "Expected plugin op in DYNAMO_ATEN_CONVERTERS")

    def test_real_plugin_graph_structure(self):
        """Verify plugin op appears in exported graph (module with lambda forward)."""
        @tta.export_as(impl=tta.plugin("MyCustomPlugin", "1.0", "test_namespace"))
        def custom_op(x):
            return x * 2.0

        model = nn.Module()
        model.forward = lambda x: custom_op(x)
        model.eval()
        exported = torch.export.export(model, (torch.randn(3, 3),))
        self.assertIn("torch_tensorrt_anno_plugin", str(exported.graph))
        plugin_nodes = [n for n in exported.graph.nodes if "torch_tensorrt_anno_plugin" in str(n.target)]
        self.assertGreater(len(plugin_nodes), 0)

    def test_real_plugin_with_real_attributes(self):
        """Plugin with real attribute values: eager result and graph."""
        @tta.export_as(
            impl=tta.plugin("ScalePlugin", "1.0", "test", scale_factor=2.5, offset=1.0, use_bias=True)
        )
        def scale_op(x):
            return x * 2.5 + 1.0

        model = nn.Module()
        model.forward = lambda x: scale_op(x)
        model.eval()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with torch.no_grad():
            result = model(x)
        torch.testing.assert_close(result, x * 2.5 + 1.0, atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(result[0, 0].item(), 3.5, places=5)
        self.assertAlmostEqual(result[1, 1].item(), 11.0, places=5)


if __name__ == "__main__":
    unittest.main()
