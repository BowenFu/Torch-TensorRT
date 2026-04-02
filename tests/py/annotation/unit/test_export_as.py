"""@export_as decorator behavior tests"""

import unittest

import torch
import torch.nn as nn

import torch_tensorrt.annotation as tta


class TestExportAsEager(unittest.TestCase):
    """Eager behavior unchanged for plugin and builtin decorators."""

    def test_export_as_decorator_plugin(self):
        """Test @export_as decorator with plugin."""

        @tta.export_as(impl=tta.plugin("TestPlugin", "1.0", "test_ns"))
        def my_op(x):
            return x * 2

        # Check metadata attached
        metadata = tta.get_annotation_metadata(my_op)
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata.impl, tta.RegistryPluginSpec)
        self.assertEqual(metadata.impl.name, "TestPlugin")

    def test_export_as_decorator_builtin(self):
        """Test @export_as decorator with builtin."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def my_relu(x):
            return torch.relu(x)

        # Check metadata attached
        metadata = tta.get_annotation_metadata(my_relu)
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata.impl, tta.BuiltinSpec)
        self.assertEqual(metadata.impl.add_name, "add_activation")

    def test_eager_behavior_unchanged_plugin(self):
        """Test decorated function works identically in eager mode (plugin)."""

        def reference_impl(x):
            return x * 2

        @tta.export_as(impl=tta.plugin("TestPlugin", "1.0", "test_ns"))
        def annotated_impl(x):
            return x * 2

        x = torch.randn(10, 20)
        expected = reference_impl(x)
        result = annotated_impl(x)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_eager_behavior_unchanged_builtin(self):
        """Test decorated function works identically in eager mode (builtin)."""

        def reference_impl(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def annotated_impl(x):
            return torch.relu(x)

        x = torch.randn(10, 20)
        expected = reference_impl(x)
        result = annotated_impl(x)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_annotation_preserves_eager_behavior(self):
        """Test that annotation doesn't break normal PyTorch execution."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def my_relu(x):
            return torch.relu(x)

        x = torch.randn(5, 5)
        result = my_relu(x)

        # Should still execute correctly in eager mode
        expected = torch.relu(x)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestExportAsGraph(unittest.TestCase):
    """Export produces correct leaf ops in graph (plugin, builtin, custom plugin)."""

    def test_export_captures_builtin_boundary(self):
        """Test that export produces builtin boundary op in graph."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def my_relu(x):
            return torch.relu(x)

        class Model(torch.nn.Module):
            def forward(self, x):
                return my_relu(x)

        model = Model()
        x = torch.randn(5, 5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_builtin", graph_str)
        self.assertIn("add_activation", graph_str)

    def test_export_captures_plugin_boundary(self):
        """Test that export produces plugin boundary op in graph."""

        @tta.export_as(impl=tta.plugin("TestPlugin", "1.0", "test_ns"))
        def my_op(x):
            return x * 2

        class Model(torch.nn.Module):
            def forward(self, x):
                return my_op(x)

        model = Model()
        x = torch.randn(5, 5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_plugin", graph_str)

    def test_simple_relu_model_export(self):
        """Test export of model with annotated ReLU."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def custom_relu(x):
            return torch.relu(x)

        class SimpleModel(nn.Module):
            def forward(self, x):
                return custom_relu(x)

        model = SimpleModel()
        x = torch.randn(3, 5)

        eager_out = model(x)
        expected = torch.relu(x)
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))

        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_builtin", graph_str)
        self.assertIn("add_activation", graph_str)

    def test_simple_plugin_export(self):
        """Test export of model with annotated plugin."""

        @tta.export_as(impl=tta.plugin("TestPlugin", "1.0", "test_namespace"))
        def custom_op(x):
            return x * 2

        class PluginModel(nn.Module):
            def forward(self, x):
                return custom_op(x)

        model = PluginModel()
        x = torch.randn(5, 5)

        eager_out = model(x)
        expected = x * 2
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)

        self.assertIn("torch_tensorrt_anno_plugin", graph_str)

    def test_plugin_with_attributes(self):
        """Test plugin with custom attributes."""

        @tta.export_as(
            impl=tta.plugin("AttrPlugin", "1.0", "test_ns", alpha=0.5, beta=1.0)
        )
        def op_with_attrs(x):
            return x * 0.5 + 1.0

        class AttrModel(nn.Module):
            def forward(self, x):
                return op_with_attrs(x)

        model = AttrModel()
        x = torch.randn(3, 3)

        eager_out = model(x)
        expected = x * 0.5 + 1.0
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_plugin", graph_str)


class TestExportAsMixed(unittest.TestCase):
    """Models with both builtin and plugin, multiple annotations, nested calls, control flow."""

    def test_multiple_builtin_layers_export(self):
        """Test model with multiple different builtin annotations."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def add(x, y):
            return x + y

        class MultiLayerModel(nn.Module):
            def forward(self, x):
                x = relu(x)
                x = add(x, torch.ones_like(x))
                return x

        model = MultiLayerModel()
        x = torch.randn(4, 4)

        eager_out = model(x)
        expected = torch.relu(x) + torch.ones_like(x)
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)

        self.assertIn("torch_tensorrt_anno_builtin", graph_str)
        self.assertIn("add_activation", graph_str)
        self.assertIn("add_elementwise", graph_str)

    def test_builtin_with_nn_module(self):
        """Test builtin annotation mixed with standard nn.Module layers."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def custom_relu(x):
            return torch.relu(x)

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                x = self.linear(x)
                x = custom_relu(x)
                return x

        model = MixedModel()
        x = torch.randn(2, 10)

        eager_out = model(x)
        self.assertEqual(eager_out.shape, (2, 10))

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)

        self.assertIn("torch_tensorrt_anno_builtin", graph_str)

    def test_multiple_plugin_callsites(self):
        """Test same plugin used multiple times in model."""

        @tta.export_as(impl=tta.plugin("ReusablePlugin", "1.0", "test"))
        def reusable_op(x):
            return x + 1

        class MultiCallModel(nn.Module):
            def forward(self, x):
                x = reusable_op(x)
                x = reusable_op(x)
                return x

        model = MultiCallModel()
        x = torch.randn(2, 2)

        eager_out = model(x)
        expected = x + 2
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)

        self.assertIn("torch_tensorrt_anno_plugin", graph_str)
        self.assertGreaterEqual(graph_str.count("torch_tensorrt_anno_plugin"), 2)

    def test_builtin_and_plugin_in_same_model(self):
        """Test model with both builtin and plugin annotations."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.plugin("CustomPlugin", "1.0", "test"))
        def custom(x):
            return x * 2

        class MixedModel(nn.Module):
            def forward(self, x):
                x = relu(x)
                x = custom(x)
                return x

        model = MixedModel()
        x = torch.randn(4, 4)

        eager_out = model(x)
        expected = torch.relu(x) * 2
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)

        self.assertIn("torch_tensorrt_anno_builtin", graph_str)
        self.assertIn("torch_tensorrt_anno_plugin", graph_str)

    def test_complex_model_with_multiple_annotations(self):
        """Test complex model with multiple annotated operations."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def add(x, y):
            return x + y

        @tta.export_as(impl=tta.plugin("Plugin1", "1.0", "test"))
        def plugin1(x):
            return x * 1.5

        @tta.export_as(impl=tta.plugin("Plugin2", "1.0", "test"))
        def plugin2(x):
            return x + 0.5

        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, x):
                x = self.linear(x)
                x = relu(x)
                x = plugin1(x)
                ones = torch.ones_like(x)
                x = add(x, ones)
                x = plugin2(x)
                return x

        model = ComplexModel()
        x = torch.randn(2, 8)

        eager_out = model(x)
        self.assertEqual(eager_out.shape, (2, 8))

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)

        self.assertIn("torch_tensorrt_anno_builtin", graph_str)
        self.assertIn("torch_tensorrt_anno_plugin", graph_str)

    def test_nested_annotated_calls(self):
        """Test nested annotated function calls."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def add(x, y):
            return x + y

        class NestedModel(nn.Module):
            def forward(self, x):
                return relu(add(x, x))

        model = NestedModel()
        x = torch.randn(5, 5)

        eager_out = model(x)
        expected = torch.relu(x + x)
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        graph_str = str(exported.graph)

        self.assertIn("add_activation", graph_str)
        self.assertIn("add_elementwise", graph_str)

    def test_multiple_annotations_in_model(self):
        """Test model with multiple different annotations."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def act(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.plugin("CustomOp", "1.0", "test"))
        def custom(x):
            return x * 2

        class Model(nn.Module):
            def forward(self, x):
                x = act(x)
                x = custom(x)
                return x

        model = Model()
        x = torch.randn(3, 3)

        out = model(x)
        expected = torch.relu(x) * 2
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x,))
        graph = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_builtin", graph)
        self.assertIn("torch_tensorrt_anno_plugin", graph)


class TestExportAsEdgeCases(unittest.TestCase):
    """Multiple input arity, annotated ops in control flow."""

    def test_annotated_op_with_multiple_inputs(self):
        """Test annotated operation with multiple tensor inputs."""

        @tta.export_as(impl=tta.builtin("add_elementwise", op=0))
        def elementwise_add(x, y):
            return x + y

        class MultiInputModel(nn.Module):
            def forward(self, x, y):
                return elementwise_add(x, y)

        model = MultiInputModel()
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        eager_out = model(x, y)
        expected = x + y
        torch.testing.assert_close(eager_out, expected, atol=1e-5, rtol=1e-5)

        exported = torch.export.export(model, (x, y))
        graph_str = str(exported.graph)
        self.assertIn("torch_tensorrt_anno_builtin", graph_str)
        self.assertIn("add_elementwise", graph_str)

    def test_annotated_op_in_control_flow(self):
        """Test annotated operation inside control flow."""

        @tta.export_as(impl=tta.builtin("add_activation", type=0))
        def relu(x):
            return torch.relu(x)

        class ControlFlowModel(nn.Module):
            def forward(self, x, use_relu):
                if use_relu:
                    return relu(x)
                else:
                    return x

        model = ControlFlowModel()
        x = torch.randn(4, 4)

        out_with_relu = model(x, True)
        out_without_relu = model(x, False)

        torch.testing.assert_close(out_with_relu, torch.relu(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_without_relu, x, atol=1e-5, rtol=1e-5)

    def test_self_attr_missing_attribute_raises(self):
        """self_attr with a nonexistent attribute path raises AttributeError at trace time."""

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op(self, x, w=tta.self_attr("nonexistent.weight")):
            return x * w

        class M(nn.Module):
            def forward(self, x):
                return op(self, x)

        m = M()
        x = torch.randn(4)
        with self.assertRaises(AttributeError):
            torch.export.export(m, (x,))


class TestExportAsName(unittest.TestCase):
    """export_as(name=) attaches name to AnnotationMetadata."""

    def test_name_stored_in_metadata(self):
        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"), name="my_op")
        def op(x):
            return x * 2
        md = tta.get_annotation_metadata(op)
        self.assertEqual(md.name, "my_op")

    def test_name_defaults_to_none(self):
        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op(x):
            return x * 2
        md = tta.get_annotation_metadata(op)
        self.assertIsNone(md.name)

    def test_name_does_not_affect_eager_behavior(self):
        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"), name="labeled")
        def op(x):
            return x * 3
        x = torch.randn(4)
        torch.testing.assert_close(op(x), x * 3)


class TestDiagnosticErrors(unittest.TestCase):
    """Stage-labeled diagnostics: verify TTADiagnosticError fields on failure paths."""

    # ------------------------------------------------------------------
    # Helper: minimal mock TRT context whose .net has no add_* methods
    # ------------------------------------------------------------------
    class _MockNet:
        """Fake trt.INetworkDefinition with no add_* methods."""
        pass

    def _MockCtx(self):
        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.net = TestDiagnosticErrors._MockNet()
        return ctx

    # ------------------------------------------------------------------
    # TTADiagnosticError base class
    # ------------------------------------------------------------------

    def test_tta_diagnostic_error_is_runtime_error(self):
        """TTADiagnosticError is a RuntimeError subclass."""
        err = tta.TTADiagnosticError("msg", stage="lowering")
        self.assertIsInstance(err, RuntimeError)

    def test_tta_diagnostic_error_fields(self):
        """TTADiagnosticError stores stage, leaf_op, impl_id."""
        err = tta.TTADiagnosticError(
            "test", stage="lowering", leaf_op="ns::op", impl_id="MyPlugin v1"
        )
        self.assertEqual(err.stage, "lowering")
        self.assertEqual(err.leaf_op, "ns::op")
        self.assertEqual(err.impl_id, "MyPlugin v1")

    def test_tta_diagnostic_error_message_contains_stage(self):
        """Error message includes the stage label."""
        err = tta.TTADiagnosticError("bad thing", stage="export")
        self.assertIn("export", str(err).lower())

    # ------------------------------------------------------------------
    # TTAPluginError subclass hierarchy
    # ------------------------------------------------------------------

    def test_plugin_error_is_diagnostic_error(self):
        """TTAPluginError is a subclass of TTADiagnosticError."""
        self.assertTrue(issubclass(tta.TTAPluginError, tta.TTADiagnosticError))

    def test_builtin_error_is_diagnostic_error(self):
        """TTABuiltinError is a subclass of TTADiagnosticError."""
        self.assertTrue(issubclass(tta.TTABuiltinError, tta.TTADiagnosticError))

    # ------------------------------------------------------------------
    # Plugin lowering: "plugin not found" path
    # ------------------------------------------------------------------

    def test_plugin_lowering_error_has_stage_lowering(self):
        """lower_plugin raises TTAPluginError with stage='lowering' when plugin not in registry (require=True)."""
        from torch_tensorrt.annotation._plugin_lowering import lower_plugin
        from torch_tensorrt.annotation._specs import RegistryPluginSpec

        spec = RegistryPluginSpec(name="NonExistentPlugin", version="1.0", namespace="ns_test")

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTADiagnosticError) as cm:
            lower_plugin(ctx, spec, [], name="test_layer", require=True)

        err = cm.exception
        self.assertEqual(err.stage, "lowering")

    def test_plugin_lowering_error_has_impl_id(self):
        """lower_plugin error carries non-empty impl_id."""
        from torch_tensorrt.annotation._plugin_lowering import lower_plugin
        from torch_tensorrt.annotation._specs import RegistryPluginSpec

        spec = RegistryPluginSpec(name="MissingPlugin", version="2.0", namespace="myns")

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTADiagnosticError) as cm:
            lower_plugin(ctx, spec, [], name="layer_x", require=True)

        err = cm.exception
        self.assertIsNotNone(err.impl_id)
        self.assertNotEqual(err.impl_id, "")

    def test_plugin_lowering_error_impl_id_contains_plugin_name(self):
        """lower_plugin impl_id includes the plugin name for traceability."""
        from torch_tensorrt.annotation._plugin_lowering import lower_plugin
        from torch_tensorrt.annotation._specs import RegistryPluginSpec

        spec = RegistryPluginSpec(name="TracedPlugin", version="1.0", namespace="trns")

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTADiagnosticError) as cm:
            lower_plugin(ctx, spec, [], name="layer_traced", require=True)

        err = cm.exception
        self.assertIn("TracedPlugin", err.impl_id)

    def test_plugin_lowering_error_has_leaf_op(self):
        """lower_plugin error carries a non-empty leaf_op."""
        from torch_tensorrt.annotation._plugin_lowering import lower_plugin
        from torch_tensorrt.annotation._specs import RegistryPluginSpec

        spec = RegistryPluginSpec(name="AnyPlugin", version="1.0", namespace="ns")

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTADiagnosticError) as cm:
            lower_plugin(ctx, spec, [], name="my_leaf_layer", require=True)

        err = cm.exception
        self.assertIsNotNone(err.leaf_op)
        self.assertNotEqual(err.leaf_op, "")

    def test_plugin_lowering_error_has_impl_id(self):
        """TTAPluginError exposes the plugin name via impl_id."""
        from torch_tensorrt.annotation._plugin_lowering import lower_plugin
        from torch_tensorrt.annotation._specs import RegistryPluginSpec

        spec = RegistryPluginSpec(name="OldPlugin", version="1.0", namespace="ns")

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTAPluginError) as cm:
            lower_plugin(ctx, spec, [], name="layer_bc", require=True)

        err = cm.exception
        self.assertIsNotNone(err.impl_id)
        self.assertIn("OldPlugin", err.impl_id)

    # ------------------------------------------------------------------
    # Builtin lowering: "method not found" path
    # ------------------------------------------------------------------

    def test_builtin_lowering_error_has_stage_lowering(self):
        """lower_builtin raises TTABuiltinError with stage='lowering' when add_* absent."""
        from torch_tensorrt.annotation._builtin_lowering import lower_builtin
        from torch_tensorrt.annotation._specs import BuiltinSpec

        spec = BuiltinSpec(add_name="add_activation", kwargs={"type": 0})

        ctx = self._MockCtx()  # _MockNet has no add_activation
        with self.assertRaises(tta.TTADiagnosticError) as cm:
            lower_builtin(ctx, spec, [], name="builtin_layer")

        err = cm.exception
        self.assertEqual(err.stage, "lowering")

    def test_builtin_lowering_error_has_impl_id(self):
        """lower_builtin error carries non-empty impl_id."""
        from torch_tensorrt.annotation._builtin_lowering import lower_builtin
        from torch_tensorrt.annotation._specs import BuiltinSpec

        spec = BuiltinSpec(add_name="add_activation", kwargs={"type": 0})

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTADiagnosticError) as cm:
            lower_builtin(ctx, spec, [], name="builtin_layer2")

        err = cm.exception
        self.assertIsNotNone(err.impl_id)
        self.assertNotEqual(err.impl_id, "")

    def test_builtin_lowering_error_has_leaf_op(self):
        """lower_builtin error carries a non-empty leaf_op."""
        from torch_tensorrt.annotation._builtin_lowering import lower_builtin
        from torch_tensorrt.annotation._specs import BuiltinSpec

        spec = BuiltinSpec(add_name="add_elementwise", kwargs={"op": 0})

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTADiagnosticError) as cm:
            lower_builtin(ctx, spec, [], name="elem_layer")

        err = cm.exception
        self.assertIsNotNone(err.leaf_op)
        self.assertNotEqual(err.leaf_op, "")

    def test_builtin_lowering_error_has_impl_id(self):
        """TTABuiltinError exposes the add_* method name via impl_id."""
        from torch_tensorrt.annotation._builtin_lowering import lower_builtin
        from torch_tensorrt.annotation._specs import BuiltinSpec

        spec = BuiltinSpec(add_name="add_activation", kwargs={"type": 0})

        ctx = self._MockCtx()
        with self.assertRaises(tta.TTABuiltinError) as cm:
            lower_builtin(ctx, spec, [], name="layer_bc2")

        err = cm.exception
        self.assertIsNotNone(err.impl_id)
        self.assertIn("add_activation", err.impl_id)

    # ------------------------------------------------------------------
    # Export-stage: self_attr unresolved
    # ------------------------------------------------------------------

    def test_export_stage_error_self_attr_unresolved(self):
        """export_as wrapper raises TTADiagnosticError(stage='export') for unresolved self_attr."""

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op(self, x, w=tta.self_attr("missing_weight")):
            return x * w

        class M(torch.nn.Module):
            def forward(self, x):
                return op(self, x)

        m = M()
        x = torch.randn(4)

        with self.assertRaises((tta.TTADiagnosticError, AttributeError, RuntimeError)):
            torch.export.export(m, (x,))

    # ------------------------------------------------------------------
    # Public API: TTADiagnosticError is accessible from tta namespace
    # ------------------------------------------------------------------

    def test_diagnostic_error_exported_from_tta(self):
        """TTADiagnosticError is importable from torch_tensorrt.annotation."""
        self.assertTrue(hasattr(tta, "TTADiagnosticError"))
        self.assertTrue(hasattr(tta, "TTAPluginError"))
        self.assertTrue(hasattr(tta, "TTABuiltinError"))

    def test_diagnostic_error_in_all(self):
        """TTADiagnosticError is listed in tta.__all__."""
        self.assertIn("TTADiagnosticError", tta.__all__)
        self.assertIn("TTAPluginError", tta.__all__)
        self.assertIn("TTABuiltinError", tta.__all__)


class TestRequireFlag(unittest.TestCase):
    """Tests for export_as(require=...) per-boundary strictness flag."""

    class _MockNet:
        """Fake trt.INetworkDefinition with no plugin registry methods."""
        pass

    class _MockCtx:
        def __init__(self):
            self.net = TestRequireFlag._MockNet()

    def test_require_false_no_raise_on_missing_plugin(self):
        """require=False: lower_plugin returns None instead of raising when plugin not found."""
        from unittest.mock import MagicMock, patch
        from torch_tensorrt.annotation._plugin_lowering import lower_plugin
        from torch_tensorrt.annotation._specs import RegistryPluginSpec

        spec = RegistryPluginSpec(name="UnregisteredPlugin", version="1.0", namespace="test_ns")
        ctx = self._MockCtx()

        mock_registry = MagicMock()
        mock_registry.get_plugin_creator.return_value = None
        mock_registry.get_creator.return_value = None

        with patch(
            "torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry",
            return_value=mock_registry,
        ), patch(
            "torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version",
            return_value=None,
        ):
            result = lower_plugin(ctx, spec, [], name="test_layer", require=False)

        self.assertIsNone(result)

    def test_require_true_raises_on_missing_plugin(self):
        """require=True: lower_plugin raises TTAPluginError with plugin name in message."""
        from unittest.mock import MagicMock, patch
        from torch_tensorrt.annotation import TTAPluginError
        from torch_tensorrt.annotation._plugin_lowering import lower_plugin
        from torch_tensorrt.annotation._specs import RegistryPluginSpec

        spec = RegistryPluginSpec(
            name="UnregisteredStrictPlugin", version="1.0", namespace="test_ns"
        )
        ctx = self._MockCtx()

        mock_registry = MagicMock()
        mock_registry.get_plugin_creator.return_value = None
        mock_registry.get_creator.return_value = None

        with patch(
            "torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry",
            return_value=mock_registry,
        ), patch(
            "torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version",
            return_value=None,
        ):
            with self.assertRaises(TTAPluginError) as cm:
                lower_plugin(ctx, spec, [], name="strict_layer", require=True)

        self.assertIn("UnregisteredStrictPlugin", str(cm.exception))

    def test_require_false_is_default(self):
        """export_as(impl=...) without require arg behaves identically to require=False."""

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op_default(x):
            return x * 2

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"), require=False)
        def op_explicit_false(x):
            return x * 2

        md_default = tta.get_annotation_metadata(op_default)
        md_explicit = tta.get_annotation_metadata(op_explicit_false)

        self.assertFalse(md_default.require)
        self.assertFalse(md_explicit.require)
        self.assertEqual(md_default.require, md_explicit.require)

    def test_require_true_stored_in_metadata(self):
        """export_as(require=True) stores require=True in AnnotationMetadata."""

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"), require=True)
        def op(x):
            return x * 2

        md = tta.get_annotation_metadata(op)
        self.assertTrue(md.require)

    def test_require_does_not_affect_eager_behavior(self):
        """require=True does not change the eager-mode result of the decorated function."""

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"), require=True)
        def op(x):
            return x * 3

        x = torch.randn(4)
        torch.testing.assert_close(op(x), x * 3)


if __name__ == "__main__":
    unittest.main()
