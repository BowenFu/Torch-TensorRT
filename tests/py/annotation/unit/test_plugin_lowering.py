"""
Unit tests for TTA plugin lowering to TensorRT.

Tests the _plugin_lowering module that converts RegistryPluginSpec + TRT tensors
to TensorRT plugin layers.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import tensorrt as trt
import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation import TTAPluginError
from torch_tensorrt.annotation._plugin_lowering import (
    encode_attrs_to_plugin_fields,
    lower_plugin,
)
from torch_tensorrt.annotation._specs import RegistryPluginSpec


class TestPluginFieldEncoding(unittest.TestCase):
    """Tests for encoding attributes to PluginFieldCollection."""

    def test_encode_int_attr(self):
        """Test encoding integer attribute."""
        attrs = {"my_int": 42}
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "my_int")
        self.assertEqual(fields[0].type, trt.PluginFieldType.INT32)
        np.testing.assert_array_equal(fields[0].data, np.array([42], dtype=np.int32))

    def test_encode_float_attr(self):
        """Test encoding float attribute."""
        attrs = {"my_float": 3.14}
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "my_float")
        self.assertEqual(fields[0].type, trt.PluginFieldType.FLOAT32)
        np.testing.assert_array_almost_equal(
            fields[0].data, np.array([3.14], dtype=np.float32)
        )

    def test_encode_bool_attr(self):
        """Test encoding boolean attribute."""
        attrs = {"my_bool": True}
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "my_bool")
        self.assertEqual(fields[0].type, trt.PluginFieldType.INT32)
        np.testing.assert_array_equal(fields[0].data, np.array([1], dtype=np.int32))

    def test_encode_string_attr(self):
        """Test encoding string attribute."""
        attrs = {"my_string": "hello"}
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "my_string")
        self.assertEqual(fields[0].type, trt.PluginFieldType.CHAR)
        self.assertIsNotNone(fields[0].data)
        self.assertGreater(len(fields[0].data), 0)

    def test_encode_list_attr(self):
        """Test encoding list/tuple attribute."""
        attrs = {"my_list": [1, 2, 3]}
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "my_list")
        self.assertEqual(fields[0].type, trt.PluginFieldType.INT32)
        self.assertIsNotNone(fields[0].data)
        self.assertEqual(len(fields[0].data), 3)

    def test_encode_numpy_array_attr(self):
        """Test encoding numpy array attribute."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        attrs = {"my_array": arr}
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "my_array")
        self.assertEqual(fields[0].type, trt.PluginFieldType.FLOAT32)
        self.assertIsNotNone(fields[0].data)
        self.assertEqual(len(fields[0].data), 3)

    def test_encode_multiple_attrs(self):
        """Test encoding multiple attributes of different types."""
        attrs = {
            "int_val": 10,
            "float_val": 2.5,
            "bool_val": False,
            "str_val": "test",
        }
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 4)
        field_names = {f.name for f in fields}
        self.assertEqual(field_names, {"int_val", "float_val", "bool_val", "str_val"})

    def test_encode_unknown_type_skipped(self):
        """Test that unknown attribute types are skipped."""
        attrs = {
            "known": 42,
            "unknown": object(),
        }
        fields = encode_attrs_to_plugin_fields(attrs)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "known")


class TestPluginLowering(unittest.TestCase):
    """Tests for plugin lowering to TensorRT."""

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.net = MagicMock()

    @patch("torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version", return_value=None)
    @patch("torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry")
    def test_lower_plugin_success(self, mock_get_registry, _mock_find):
        """Test successful plugin lowering via the V2 path."""
        mock_registry = MagicMock()
        mock_creator = MagicMock()
        mock_plugin = MagicMock()
        mock_layer = MagicMock()
        mock_layer.num_outputs = 1
        mock_output = MagicMock()

        mock_get_registry.return_value = mock_registry
        mock_registry.get_creator.return_value = None  # force V2 path
        mock_registry.get_plugin_creator.return_value = mock_creator
        mock_creator.create_plugin.return_value = mock_plugin
        mock_layer.get_output.return_value = mock_output
        self.mock_ctx.net.add_plugin_v2.return_value = mock_layer

        mock_input = MagicMock()
        spec = RegistryPluginSpec(
            name="TestPlugin", version="1.0", namespace="test_ns", attrs={"alpha": 0.5}
        )

        result = lower_plugin(self.mock_ctx, spec, [mock_input], "test_plugin")

        mock_registry.get_plugin_creator.assert_called_once_with(
            "TestPlugin", "1.0", "test_ns"
        )
        mock_creator.create_plugin.assert_called_once()
        self.mock_ctx.net.add_plugin_v2.assert_called_once()
        call_args = self.mock_ctx.net.add_plugin_v2.call_args[0]
        self.assertIn(mock_input, call_args[0])
        self.assertEqual(call_args[1], mock_plugin)
        self.assertEqual(mock_layer.name, "test_plugin")
        self.assertEqual(result, mock_output)

    @patch("torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version", return_value=None)
    @patch("torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry")
    def test_lower_plugin_not_found(self, mock_get_registry, _mock_find):
        """Test error when plugin not found in registry (V2 and V3 both miss)."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_registry.get_plugin_creator.return_value = None
        mock_registry.get_creator.return_value = None

        mock_input = MagicMock()
        spec = RegistryPluginSpec(
            name="NonexistentPlugin", version="1.0", namespace="test_ns"
        )

        with self.assertRaises(TTAPluginError) as ctx:
            lower_plugin(self.mock_ctx, spec, [mock_input], "test_error")

        error_msg = str(ctx.exception)
        self.assertIn("Plugin not found", error_msg)
        self.assertIn("NonexistentPlugin", error_msg)

    @patch("torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version", return_value=None)
    @patch("torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry")
    def test_lower_plugin_creation_fails(self, mock_get_registry, _mock_find):
        """Test error when plugin creation fails on all paths."""
        mock_registry = MagicMock()
        mock_creator = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_registry.get_plugin_creator.return_value = mock_creator
        mock_creator.create_plugin.side_effect = RuntimeError("Plugin init failed")
        mock_registry.get_creator.return_value = None

        mock_input = MagicMock()
        spec = RegistryPluginSpec(name="TestPlugin", version="1.0", namespace="test_ns")

        with self.assertRaises((TTAPluginError, RuntimeError)):
            lower_plugin(self.mock_ctx, spec, [mock_input], "test_error")

    @patch("torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version", return_value=None)
    @patch("torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry")
    def test_lower_plugin_add_layer_fails(self, mock_get_registry, _mock_find):
        """Test error when adding plugin layer to network fails on all paths."""
        mock_registry = MagicMock()
        mock_creator = MagicMock()
        mock_plugin = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_registry.get_plugin_creator.return_value = mock_creator
        mock_creator.create_plugin.return_value = mock_plugin
        self.mock_ctx.net.add_plugin_v2.side_effect = RuntimeError("Network error")
        self.mock_ctx.net.add_plugin_v3.side_effect = RuntimeError("Network error")
        mock_registry.get_creator.return_value = None

        mock_input = MagicMock()
        spec = RegistryPluginSpec(name="TestPlugin", version="1.0", namespace="test_ns")

        with self.assertRaises((TTAPluginError, RuntimeError)):
            lower_plugin(self.mock_ctx, spec, [mock_input], "test_error")

    @patch("torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version", return_value=None)
    @patch("torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry")
    def test_lower_plugin_multiple_inputs(self, mock_get_registry, _mock_find):
        """Test plugin lowering with multiple input tensors via the V2 path."""
        mock_registry = MagicMock()
        mock_creator = MagicMock()
        mock_plugin = MagicMock()
        mock_layer = MagicMock()
        mock_layer.num_outputs = 1
        mock_output = MagicMock()

        mock_get_registry.return_value = mock_registry
        mock_registry.get_creator.return_value = None  # force V2 path
        mock_registry.get_plugin_creator.return_value = mock_creator
        mock_creator.create_plugin.return_value = mock_plugin
        mock_layer.get_output.return_value = mock_output
        self.mock_ctx.net.add_plugin_v2.return_value = mock_layer

        mock_input1 = MagicMock()
        mock_input2 = MagicMock()
        mock_input3 = MagicMock()
        spec = RegistryPluginSpec(name="MultiInputPlugin", version="1.0", namespace="test_ns")

        result = lower_plugin(
            self.mock_ctx, spec, [mock_input1, mock_input2, mock_input3], "test_multi"
        )

        call_args = self.mock_ctx.net.add_plugin_v2.call_args[0][0]
        self.assertEqual(len(call_args), 3)
        self.assertIn(mock_input1, call_args)
        self.assertIn(mock_input2, call_args)
        self.assertIn(mock_input3, call_args)

    @patch("torch_tensorrt.annotation._plugin_lowering._find_creator_by_name_version", return_value=None)
    @patch("torch_tensorrt.annotation._plugin_lowering.trt.get_plugin_registry")
    def test_lower_plugin_multiple_outputs(self, mock_get_registry, _mock_find):
        """Test plugin lowering with multiple outputs via the V2 path."""
        mock_registry = MagicMock()
        mock_creator = MagicMock()
        mock_plugin = MagicMock()
        mock_layer = MagicMock()
        mock_layer.num_outputs = 2
        mock_output0 = MagicMock()
        mock_output1 = MagicMock()
        mock_layer.get_output.side_effect = [mock_output0, mock_output1]

        mock_get_registry.return_value = mock_registry
        mock_registry.get_creator.return_value = None  # force V2 path
        mock_registry.get_plugin_creator.return_value = mock_creator
        mock_creator.create_plugin.return_value = mock_plugin
        self.mock_ctx.net.add_plugin_v2.return_value = mock_layer

        mock_input = MagicMock()
        spec = RegistryPluginSpec(name="MultiOutputPlugin", version="1.0", namespace="test_ns")

        result = lower_plugin(self.mock_ctx, spec, [mock_input], "test_multi_out")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_output0)
        self.assertEqual(result[1], mock_output1)


if __name__ == "__main__":
    unittest.main()
