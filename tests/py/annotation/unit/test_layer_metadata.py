"""Unit tests for TTA layer metadata (set_tta_layer_metadata)."""

import unittest
from unittest.mock import MagicMock

from torch_tensorrt.annotation._layer_metadata import (
    parse_tta_layer_metadata,
    set_tta_layer_metadata,
)


class TestSetTTALayerMetadata(unittest.TestCase):
    def test_sets_metadata_when_layer_has_metadata_attribute(self):
        layer = MagicMock()
        set_tta_layer_metadata(layer, "builtin", "add_activation", "relu")
        payload = parse_tta_layer_metadata(layer.metadata)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["backend"], "builtin")
        self.assertEqual(payload["plugin_name"], "add_activation")
        self.assertEqual(payload["torch_op"], "relu")

    def test_calls_set_metadata_when_layer_has_set_metadata_method(self):
        class LayerWithSetMetadata:
            set_metadata = MagicMock()

        layer = LayerWithSetMetadata()
        set_tta_layer_metadata(layer, "plugin", "InstanceNormalization_TRT", "instnorm")
        layer.set_metadata.assert_called_once()
        (arg,) = layer.set_metadata.call_args[0]
        payload = parse_tta_layer_metadata(arg)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["backend"], "plugin")
        self.assertEqual(payload["plugin_name"], "InstanceNormalization_TRT")
        self.assertEqual(payload["torch_op"], "instnorm")

    def test_custom_plugin_metadata_content(self):
        layer = MagicMock()
        set_tta_layer_metadata(layer, "triton", "scale", "scale_op")
        payload = parse_tta_layer_metadata(layer.metadata)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["backend"], "triton")
        self.assertEqual(payload["plugin_name"], "scale")
        self.assertEqual(payload["torch_op"], "scale_op")

    def test_attrs_encoded_and_parsed(self):
        layer = MagicMock()
        set_tta_layer_metadata(
            layer, "triton", "swiglu_v1", "model.encoder.mlp",
            attrs={"BLOCK_M": 3, "addend": 11.0},
        )
        payload = parse_tta_layer_metadata(layer.metadata)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["backend"], "triton")
        self.assertEqual(payload["plugin_name"], "swiglu_v1")
        self.assertEqual(payload["torch_op"], "model.encoder.mlp")
        self.assertEqual(payload["attrs"]["BLOCK_M"], 3)
        self.assertAlmostEqual(payload["attrs"]["addend"], 11.0)

    def test_no_op_when_layer_has_neither_metadata_nor_set_metadata(self):
        layer = object()
        # Must not raise; layer has no metadata attribute and no set_metadata method.
        set_tta_layer_metadata(layer, "builtin", "add_activation", "relu")
        self.assertFalse(hasattr(layer, "metadata"))

    def test_empty_attrs(self):
        layer = MagicMock()
        set_tta_layer_metadata(layer, "builtin", "add_activation", "relu", attrs={})
        payload = parse_tta_layer_metadata(layer.metadata)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["attrs"], {})

    def test_fn_specs_encoded_and_parsed(self):
        from torch_tensorrt.annotation._layer_metadata import _format_tta_metadata
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        fn_specs = [("launch_add_one", {"BLOCK_SIZE": 128}), ("launch_add_one", {"BLOCK_SIZE": 256})]
        raw = _format_tta_metadata("triton", "host_kernel_abc", "model.fc", fn_specs=fn_specs)
        parsed = _parse_single_tta_segment(raw)
        self.assertIsNotNone(parsed)
        self.assertIn("fn_specs", parsed)
        self.assertEqual(len(parsed["fn_specs"]), 2)
        self.assertEqual(parsed["fn_specs"][0]["fn_name"], "launch_add_one")
        self.assertEqual(parsed["fn_specs"][0]["config"]["BLOCK_SIZE"], 128)
        self.assertEqual(parsed["fn_specs"][1]["config"]["BLOCK_SIZE"], 256)

    def test_parse_returns_none_for_empty_torch_op(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        # A well-formed tier-1 string with an empty torch_op should return None.
        raw = "tta triton:my_kernel attrs: torch_op:"
        result = _parse_single_tta_segment(raw)
        self.assertIsNone(result)
