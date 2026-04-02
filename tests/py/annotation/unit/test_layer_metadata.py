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


class TestNodeTorchOpPath(unittest.TestCase):
    def test_returns_name_when_no_stack(self):
        from torch_tensorrt.annotation._layer_metadata import tta_observe_perf_torch_op_path
        node = MagicMock()
        node.name = "my_node"
        node.meta = {}
        self.assertEqual(tta_observe_perf_torch_op_path(node), "my_node")

    def test_uses_nn_module_stack(self):
        from torch_tensorrt.annotation._layer_metadata import tta_observe_perf_torch_op_path
        node = MagicMock()
        node.name = "act"
        node.meta = {"nn_module_stack": {"mod": ("encoder.mlp", object)}}
        result = tta_observe_perf_torch_op_path(node)
        self.assertEqual(result, "encoder.mlp/act")

    def test_uses_deepest_nonempty_qname(self):
        from torch_tensorrt.annotation._layer_metadata import tta_observe_perf_torch_op_path
        node = MagicMock()
        node.name = "fc"
        node.meta = {"nn_module_stack": {
            "a": ("outer", object),
            "b": ("outer.inner", object),
        }}
        result = tta_observe_perf_torch_op_path(node)
        self.assertEqual(result, "outer.inner/fc")

    def test_falls_back_when_stack_empty_dict(self):
        from torch_tensorrt.annotation._layer_metadata import tta_observe_perf_torch_op_path
        node = MagicMock()
        node.name = "x"
        node.meta = {"nn_module_stack": {}}
        self.assertEqual(tta_observe_perf_torch_op_path(node), "x")


class TestTacticsString(unittest.TestCase):
    def test_empty_specs_returns_empty_string(self):
        from torch_tensorrt.annotation._layer_metadata import tactics_string
        self.assertEqual(tactics_string([]), "")

    def test_single_triton_spec_no_configs(self):
        import torch_tensorrt.annotation as tta
        from torch_tensorrt.annotation._layer_metadata import tactics_string

        def launch_fn(x, out):
            pass

        spec = tta.triton(launch_fn)
        result = tactics_string([spec])
        self.assertIn("triton", result)
        self.assertIn("launch_fn", result)
        self.assertTrue(result.startswith("1:"))

    def test_single_triton_spec_with_configs(self):
        import torch_tensorrt.annotation as tta
        from torch_tensorrt.annotation._layer_metadata import tactics_string

        def launch_fn(x, out):
            pass

        spec = tta.triton(launch_fn, configs=[{"BLOCK": 64}, {"BLOCK": 128}])
        result = tactics_string([spec])
        parts = result.split("|")
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].startswith("1:"))
        self.assertTrue(parts[1].startswith("2:"))

    def test_cutile_and_cutedsl_backends(self):
        import torch_tensorrt.annotation as tta
        from torch_tensorrt.annotation._layer_metadata import tactics_string

        def fn(x, out):
            pass

        result = tactics_string([tta.cutile(fn), tta.cutedsl(fn)])
        self.assertIn("cutile", result)
        self.assertIn("cutedsl", result)

    def test_unknown_spec_type_uses_class_name(self):
        from torch_tensorrt.annotation._layer_metadata import tactics_string

        class MyCustomSpec:
            configs = [{"X": 1}]
            launch_fn = lambda: None
            launch_fn.__name__ = "my_fn"

        result = tactics_string([MyCustomSpec()])
        self.assertIn("mycustomspec", result)

    def test_indices_are_monotonically_increasing(self):
        import torch_tensorrt.annotation as tta
        from torch_tensorrt.annotation._layer_metadata import tactics_string

        def fn(x, out):
            pass

        spec = tta.triton(fn, configs=[{"B": 1}, {"B": 2}, {"B": 3}])
        parts = tactics_string([spec]).split("|")
        indices = [int(p.split(":")[0]) for p in parts]
        self.assertEqual(indices, [1, 2, 3])


class TestValidateAttrKey(unittest.TestCase):
    def test_valid_key_no_raise(self):
        from torch_tensorrt.annotation._layer_metadata import _validate_attr_key
        _validate_attr_key("BLOCK_SIZE")  # should not raise

    def test_key_with_equals_raises(self):
        from torch_tensorrt.annotation._layer_metadata import _validate_attr_key
        with self.assertRaises(ValueError):
            _validate_attr_key("bad=key")

    def test_key_with_comma_raises(self):
        from torch_tensorrt.annotation._layer_metadata import _validate_attr_key
        with self.assertRaises(ValueError):
            _validate_attr_key("a,b")

    def test_key_with_colon_raises(self):
        from torch_tensorrt.annotation._layer_metadata import _validate_attr_key
        with self.assertRaises(ValueError):
            _validate_attr_key("ns:key")

    def test_key_with_pipe_raises(self):
        from torch_tensorrt.annotation._layer_metadata import _validate_attr_key
        with self.assertRaises(ValueError):
            _validate_attr_key("a|b")


class TestFormatTier2(unittest.TestCase):
    def test_format_and_parse_tier2(self):
        from torch_tensorrt.annotation._layer_metadata import (
            _format_tta_metadata_tier2,
            _parse_single_tta_segment,
        )
        raw = _format_tta_metadata_tier2("model.encoder.fc")
        result = _parse_single_tta_segment(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["torch_op"], "model.encoder.fc")
        self.assertNotIn("backend", result)


class TestParseValue(unittest.TestCase):
    def test_int(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_value
        self.assertEqual(_parse_value("42"), 42)
        self.assertIsInstance(_parse_value("42"), int)

    def test_float(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_value
        self.assertAlmostEqual(_parse_value("3.14"), 3.14)
        self.assertIsInstance(_parse_value("3.14"), float)

    def test_string_fallback(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_value
        self.assertEqual(_parse_value("hello"), "hello")
        self.assertIsInstance(_parse_value("hello"), str)


class TestParseSingleSegmentEdgeCases(unittest.TestCase):
    def test_non_tta_prefix_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        self.assertIsNone(_parse_single_tta_segment("something else"))

    def test_empty_string_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        self.assertIsNone(_parse_single_tta_segment(""))

    def test_tier2_parse(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        result = _parse_single_tta_segment("tta torch_op:my.op")
        self.assertIsNotNone(result)
        self.assertEqual(result["torch_op"], "my.op")

    def test_too_few_tokens_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        self.assertIsNone(_parse_single_tta_segment("tta triton:op attrs:"))

    def test_no_colon_in_backend_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        self.assertIsNone(_parse_single_tta_segment("tta nocolon attrs: torch_op:x"))

    def test_missing_attrs_token_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        self.assertIsNone(_parse_single_tta_segment("tta triton:op notattrs: torch_op:x"))

    def test_missing_torch_op_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        self.assertIsNone(_parse_single_tta_segment("tta triton:op attrs: notop:x"))

    def test_fn_then_no_more_tokens_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        # fn: token at idx 2, attrs at idx 3, nothing after — tok_idx(4) >= len(4)
        self.assertIsNone(_parse_single_tta_segment("tta triton:op fn:foo attrs:"))

    def test_attrs_then_no_torch_op_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        # attrs: token valid but nothing follows — tok_idx >= len(tokens)
        self.assertIsNone(_parse_single_tta_segment("tta triton:op attrs: "))

    def test_unexpected_token_between_attrs_and_torch_op(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        # unexpected token triggers debug log but parse still succeeds
        result = _parse_single_tta_segment("tta triton:op attrs: extra torch_op:x")
        self.assertIsNotNone(result)
        self.assertEqual(result["torch_op"], "x")

    def test_parse_fn_specs_empty_entry_skipped(self):
        from torch_tensorrt.annotation._layer_metadata import _parse_single_tta_segment
        # fn: value "|bar:" has an empty first entry ("") which triggers `continue`
        result = _parse_single_tta_segment("tta triton:op fn:|bar: attrs: torch_op:x")
        self.assertIsNotNone(result)
        self.assertIn("fn_specs", result)
        self.assertEqual(len(result["fn_specs"]), 1)  # empty entry skipped


class TestSegmentPriority(unittest.TestCase):
    def test_none_returns_minus_one(self):
        from torch_tensorrt.annotation._layer_metadata import _segment_priority
        self.assertEqual(_segment_priority(None), -1)

    def test_tier1_backend_returns_two(self):
        from torch_tensorrt.annotation._layer_metadata import _segment_priority
        self.assertEqual(_segment_priority({"backend": "triton", "torch_op": "x"}), 2)

    def test_unknown_backend_returns_one(self):
        from torch_tensorrt.annotation._layer_metadata import _segment_priority
        self.assertEqual(_segment_priority({"backend": "autotune", "torch_op": "x"}), 1)

    def test_tier2_no_backend_returns_zero(self):
        from torch_tensorrt.annotation._layer_metadata import _segment_priority
        self.assertEqual(_segment_priority({"torch_op": "x"}), 0)


class TestParseLayerMetadata(unittest.TestCase):
    def test_empty_string_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import parse_tta_layer_metadata
        self.assertIsNone(parse_tta_layer_metadata(""))

    def test_whitespace_only_returns_none(self):
        from torch_tensorrt.annotation._layer_metadata import parse_tta_layer_metadata
        self.assertIsNone(parse_tta_layer_metadata("   "))

    def test_fused_picks_highest_priority(self):
        from torch_tensorrt.annotation._layer_metadata import parse_tta_layer_metadata
        tier1 = "tta triton:my_op attrs: torch_op:fc"
        tier2 = "tta torch_op:fc"
        # tier1 wins
        result = parse_tta_layer_metadata(f"{tier2}\x1f{tier1}")
        self.assertIsNotNone(result)
        self.assertEqual(result["backend"], "triton")


class TestSetTTALayerMetadataAttributeError(unittest.TestCase):
    def test_attribute_error_on_metadata_set_is_suppressed(self):
        class ReadOnlyLayer:
            @property
            def metadata(self):
                return ""

            @metadata.setter
            def metadata(self, v):
                raise AttributeError("read-only")

        layer = ReadOnlyLayer()
        # Should not raise
        set_tta_layer_metadata(layer, "triton", "op", "torch_op")
