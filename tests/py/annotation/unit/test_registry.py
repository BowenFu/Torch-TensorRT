"""
Unit tests for implementation registry.

Tests the registry that tracks plugin/builtin implementations and generates
unique torch.ops identities.
"""

import unittest

from torch_tensorrt.annotation._registry import get_or_create_op_for_boundary
from torch_tensorrt.annotation import builtin, plugin


class TestRegistry(unittest.TestCase):
    """Tests for implementation registry."""

    def test_same_identity_returns_same_op(self):
        """Test that same plugin identity returns same op."""
        spec1 = plugin("TestPlugin", "1.0", "test_ns", alpha=0.5)
        spec2 = plugin("TestPlugin", "1.0", "test_ns", alpha=0.5)

        op1 = get_or_create_op_for_boundary(spec1, runtime_input_arity=1)
        op2 = get_or_create_op_for_boundary(spec2, runtime_input_arity=1)

        # Should return the same op for same identity
        self.assertEqual(str(op1), str(op2))

    def test_different_identity_returns_different_op(self):
        """Test that different identities return different ops."""
        spec1 = plugin("TestPlugin", "1.0", "test_ns", alpha=0.5)
        spec2 = plugin("TestPlugin", "2.0", "test_ns", alpha=0.5)  # Different version

        op1 = get_or_create_op_for_boundary(spec1, runtime_input_arity=1)
        op2 = get_or_create_op_for_boundary(spec2, runtime_input_arity=1)

        # Should return different ops for different identities
        self.assertNotEqual(str(op1), str(op2))

    def test_plugin_namespace(self):
        """Test that plugin ops use correct namespace."""
        spec = plugin("TestPlugin", "1.0", "test_ns")
        op = get_or_create_op_for_boundary(spec, runtime_input_arity=1)

        op_str = str(op)
        self.assertIn("torch_tensorrt_anno_plugin", op_str)

    def test_builtin_namespace(self):
        """Test that builtin ops use correct namespace."""
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        op = get_or_create_op_for_boundary(spec, runtime_input_arity=1)

        op_str = str(op)
        self.assertIn("torch_tensorrt_anno_builtin", op_str)

    def test_different_attrs_different_ops(self):
        """Test that different attrs result in different ops."""
        # For plugins, the full identity includes attrs
        # Different attrs result in different cache keys
        spec1 = plugin("TestPlugin", "1.0", "test_ns", alpha=0.5)
        spec2 = plugin("TestPlugin", "1.0", "test_ns", alpha=1.0)  # Different alpha

        op1 = get_or_create_op_for_boundary(spec1, runtime_input_arity=1)
        op2 = get_or_create_op_for_boundary(spec2, runtime_input_arity=1)

        # Should return different ops (attrs affect identity)
        self.assertNotEqual(str(op1), str(op2))

    def test_zero_input_arity(self):
        """Test op registration with zero inputs."""
        spec = builtin("add_constant", shape=(1,), weights=None)
        op = get_or_create_op_for_boundary(spec, runtime_input_arity=0)

        # Op should be created successfully
        self.assertIsNotNone(op)
        op_str = str(op)
        self.assertIn("torch_tensorrt_anno_builtin", op_str)

    def test_multiple_input_arity(self):
        """Test op registration with multiple inputs."""
        spec = builtin("add_elementwise", op=0)

        # Test with 2 inputs
        op2 = get_or_create_op_for_boundary(spec, runtime_input_arity=2)
        self.assertIsNotNone(op2)

        # Test with 3 inputs
        op3 = get_or_create_op_for_boundary(spec, runtime_input_arity=3)
        self.assertIsNotNone(op3)

        # Different arities should create different ops
        self.assertNotEqual(str(op2), str(op3))

    def test_builtin_op_caching(self):
        """Test that builtin ops are cached properly."""
        spec = builtin("add_activation", type=0)

        # First call creates the op
        op1 = get_or_create_op_for_boundary(spec, runtime_input_arity=1)

        # Second call with same spec should return cached op
        op2 = get_or_create_op_for_boundary(spec, runtime_input_arity=1)

        # Should be the exact same op (cached)
        self.assertEqual(str(op1), str(op2))

    def test_invalid_spec_type(self):
        """Test that invalid spec type raises TypeError."""
        # Invalid spec type should raise TypeError
        with self.assertRaises(TypeError):
            get_or_create_op_for_boundary("invalid_string", runtime_input_arity=1)

        with self.assertRaises(TypeError):
            get_or_create_op_for_boundary(123, runtime_input_arity=1)


if __name__ == "__main__":
    unittest.main()
