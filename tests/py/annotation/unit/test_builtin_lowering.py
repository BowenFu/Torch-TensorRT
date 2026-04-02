"""
Unit tests for TTA builtin lowering to TensorRT.

Tests the _builtin_lowering module that converts BuiltinSpec + TRT tensors
to TensorRT layers via add_* methods.
"""

import unittest
from unittest.mock import MagicMock

import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation import TTABuiltinError
from torch_tensorrt.annotation._builtin_lowering import lower_builtin
from torch_tensorrt.annotation._specs import BuiltinSpec


class TestBuiltinLowering(unittest.TestCase):
    """Tests for builtin lowering to TensorRT."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ctx = MagicMock()
        self.mock_ctx.net = MagicMock()

    def test_lower_activation_builtin(self):
        """Test lowering add_activation builtin to TRT layer."""
        mock_layer = MagicMock()
        mock_layer.num_outputs = 1
        mock_output = MagicMock()
        mock_layer.get_output.return_value = mock_output
        self.mock_ctx.net.add_activation = MagicMock(return_value=mock_layer)

        mock_input = MagicMock()
        spec = BuiltinSpec(add_name="add_activation", kwargs={"type": 0})

        result = lower_builtin(self.mock_ctx, spec, [mock_input], "test_relu")

        self.mock_ctx.net.add_activation.assert_called_once()
        self.assertEqual(mock_layer.name, "test_relu")
        self.assertEqual(result, mock_output)

    def test_lower_constant_builtin_zero_inputs(self):
        """Test lowering add_constant with zero tensor inputs."""
        mock_layer = MagicMock()
        mock_layer.num_outputs = 1
        mock_output = MagicMock()
        mock_layer.get_output.return_value = mock_output
        self.mock_ctx.net.add_constant = MagicMock(return_value=mock_layer)

        spec = BuiltinSpec(
            add_name="add_constant",
            kwargs={"shape": (1, 3, 224, 224), "weights": None},
        )

        result = lower_builtin(self.mock_ctx, spec, [], "test_constant")

        self.mock_ctx.net.add_constant.assert_called_once()
        self.assertEqual(result, mock_output)

    def test_lower_elementwise_builtin_multiple_inputs(self):
        """Test lowering add_elementwise with multiple tensor inputs."""
        mock_layer = MagicMock()
        mock_layer.num_outputs = 1
        mock_output = MagicMock()
        mock_layer.get_output.return_value = mock_output
        self.mock_ctx.net.add_elementwise = MagicMock(return_value=mock_layer)

        mock_input1 = MagicMock()
        mock_input2 = MagicMock()
        spec = BuiltinSpec(add_name="add_elementwise", kwargs={"op": 0})

        result = lower_builtin(
            self.mock_ctx, spec, [mock_input1, mock_input2], "test_add"
        )

        self.mock_ctx.net.add_elementwise.assert_called_once()
        call_args = self.mock_ctx.net.add_elementwise.call_args
        self.assertIn(mock_input1, call_args[0])
        self.assertIn(mock_input2, call_args[0])

    def test_lower_builtin_multiple_outputs(self):
        """Test lowering builtin with multiple output tensors."""
        mock_layer = MagicMock()
        mock_layer.num_outputs = 2
        mock_output0 = MagicMock()
        mock_output1 = MagicMock()
        mock_layer.get_output.side_effect = [mock_output0, mock_output1]
        self.mock_ctx.net.add_activation = MagicMock(return_value=mock_layer)

        mock_input = MagicMock()
        spec = BuiltinSpec(add_name="add_activation", kwargs={"type": 0})

        result = lower_builtin(self.mock_ctx, spec, [mock_input], "test_multi")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_output0)
        self.assertEqual(result[1], mock_output1)

    def test_lower_builtin_method_not_found(self):
        """Test error when add_* method doesn't exist."""
        mock_input = MagicMock()
        spec = BuiltinSpec(add_name="add_nonexistent", kwargs={})

        with self.assertRaises(TTABuiltinError) as ctx:
            lower_builtin(self.mock_ctx, spec, [mock_input], "test_error")

        error_msg = str(ctx.exception)
        self.assertIn("add_nonexistent", error_msg)
        self.assertIn("not found", error_msg.lower())

    def test_lower_builtin_add_method_fails(self):
        """Test that TRT errors propagate directly when add_* method fails."""
        self.mock_ctx.net.add_activation = MagicMock(
            side_effect=RuntimeError("TRT internal error")
        )

        mock_input = MagicMock()
        spec = BuiltinSpec(add_name="add_activation", kwargs={"type": 0})

        with self.assertRaises(RuntimeError) as ctx:
            lower_builtin(self.mock_ctx, spec, [mock_input], "test_error")

        self.assertIn("TRT internal error", str(ctx.exception))

    def test_lower_builtin_setattr_application(self):
        """Test that setattr is applied for leftover attributes."""
        mock_layer = MagicMock()
        mock_layer.num_outputs = 1
        mock_output = MagicMock()
        mock_layer.get_output.return_value = mock_output
        self.mock_ctx.net.add_activation = MagicMock(return_value=mock_layer)

        mock_input = MagicMock()
        spec = BuiltinSpec(
            add_name="add_activation",
            kwargs={"type": 0, "alpha": 0.1, "beta": 0.2},
        )

        lower_builtin(self.mock_ctx, spec, [mock_input], "test_setattr")

        self.mock_ctx.net.add_activation.assert_called_once()


class TestBuiltinLoweringEdgeCases(unittest.TestCase):
    """Edge case tests for builtin lowering."""

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.net = MagicMock()

    def test_lower_builtin_empty_kwargs(self):
        """Test lowering with empty kwargs spec."""
        mock_layer = MagicMock()
        mock_layer.num_outputs = 1
        mock_output = MagicMock()
        mock_layer.get_output.return_value = mock_output
        self.mock_ctx.net.add_activation = MagicMock(return_value=mock_layer)

        mock_input = MagicMock()
        spec = BuiltinSpec(add_name="add_activation", kwargs={})

        result = lower_builtin(self.mock_ctx, spec, [mock_input], "test_empty")

        self.assertIsNotNone(result)
        self.mock_ctx.net.add_activation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
