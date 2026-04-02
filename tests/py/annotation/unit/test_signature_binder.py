"""
Unit tests for SignatureModel and binder utilities.

Tests signature model and builtin spec functionality.
"""

import unittest

import torch

from torch_tensorrt.annotation._signature import (
    _extract_from_trt_doc,
    _parse_param_names_from_doc,
    get_signature_model,
)
from torch_tensorrt.annotation import builtin


class TestSignatureBinder(unittest.TestCase):
    """Tests for SignatureModel and parameter binding."""

    def test_signature_model_creation(self):
        """Test SignatureModel can be created."""
        spec = builtin("add_constant", shape=(1,), weights=None)
        sig = get_signature_model(spec)
        self.assertIsNotNone(sig)
        self.assertIsNotNone(sig.ctor_param_names)
        self.assertEqual(sig.add_name, "add_constant")

    def test_builtin_spec_kwargs(self):
        """Test BuiltinSpec correctly stores kwargs that will be passed to TRT."""
        spec = builtin("add_constant", shape=(1,), weights=None)

        self.assertEqual(spec.add_name, "add_constant")
        self.assertIn("shape", spec.kwargs)
        self.assertEqual(spec.kwargs["shape"], (1,))

    def test_builtin_spec_with_tuple(self):
        """Test BuiltinSpec preserves tuple types (important for padding, strides, etc)."""
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)

        # For kernel_shape, we expect tuple/list to be stored as-is
        self.assertIsInstance(spec.kwargs["kernel_shape"], tuple)
        self.assertEqual(spec.kwargs["kernel_shape"], (3, 3))

    def test_signature_deterministic(self):
        """Test that signature creation is deterministic (important for caching)."""
        spec = builtin("add_constant", shape=(1,), weights=None)
        sig1 = get_signature_model(spec)
        sig2 = get_signature_model(spec)

        # Should return consistent signatures
        self.assertEqual(sig1.ctor_param_names, sig2.ctor_param_names)
        self.assertEqual(sig1.add_name, sig2.add_name)

    def test_extract_from_trt_doc_method_not_found(self):
        """Test error when method doesn't exist."""
        with self.assertRaises(ValueError) as ctx:
            _extract_from_trt_doc("nonexistent_method")
        self.assertIn("not found", str(ctx.exception))

    def test_parse_param_names_empty_doc(self):
        """Test parsing with empty docstring."""
        result = _parse_param_names_from_doc("", "add_constant")
        self.assertEqual(result, tuple())

    def test_parse_param_names_no_signature(self):
        """Test parsing with docstring but no signature line."""
        doc = "This is just some text\nwithout any signature"
        result = _parse_param_names_from_doc(doc, "add_constant")
        self.assertEqual(result, tuple())

    def test_parse_param_names_fallback_search(self):
        """Test fallback search when add_name not in signature."""
        doc = "some_other_method(self, param1: int) -> None"
        result = _parse_param_names_from_doc(doc, "add_constant")
        # Should find the signature and extract param1
        self.assertEqual(result, ("param1",))

    def test_parse_param_names_no_parentheses(self):
        """Test parsing when signature has no parentheses."""
        doc = "add_constant no parens here"
        result = _parse_param_names_from_doc(doc, "add_constant")
        self.assertEqual(result, tuple())

    def test_parse_param_names_missing_close_paren(self):
        """Test parsing when signature has opening but no closing paren."""
        doc = "add_constant(self, shape: tensorrt.Dims"
        result = _parse_param_names_from_doc(doc, "add_constant")
        self.assertEqual(result, tuple())

    def test_parse_param_names_empty_params(self):
        """Test parsing when signature has empty parameters."""
        doc = "add_constant() -> IConstantLayer"
        result = _parse_param_names_from_doc(doc, "add_constant")
        self.assertEqual(result, tuple())

    def test_parse_param_names_with_defaults(self):
        """Test parsing parameters with default values."""
        doc = "add_convolution_nd(self, input: tensorrt.ITensor, num_output_maps: int, kernel_shape: tensorrt.Dims, kernel: tensorrt.Weights, bias: tensorrt.Weights = None) -> IConvolutionLayer"
        result = _parse_param_names_from_doc(doc, "add_convolution_nd")
        self.assertEqual(
            result, ("input", "num_output_maps", "kernel_shape", "kernel", "bias")
        )

    def test_parse_param_names_filters_self(self):
        """Test that self/this/cls are filtered out."""
        doc1 = "add_constant(self, shape: tensorrt.Dims) -> IConstantLayer"
        result1 = _parse_param_names_from_doc(doc1, "add_constant")
        self.assertEqual(result1, ("shape",))

        doc2 = "add_activation(this, input: tensorrt.ITensor) -> IActivationLayer"
        result2 = _parse_param_names_from_doc(doc2, "add_activation")
        self.assertEqual(result2, ("input",))

        doc3 = "add_pooling_nd(cls, input: tensorrt.ITensor) -> IPoolingLayer"
        result3 = _parse_param_names_from_doc(doc3, "add_pooling_nd")
        self.assertEqual(result3, ("input",))

    def test_parse_param_names_empty_tokens(self):
        """Test parsing handles empty tokens between commas."""
        doc = "add_constant(self,  , shape: tensorrt.Dims) -> IConstantLayer"
        result = _parse_param_names_from_doc(doc, "add_constant")
        self.assertEqual(result, ("shape",))

    def test_e2e_public_api_add_constant(self):
        """Test end-to-end public API with real TensorRT add_constant."""
        spec = builtin("add_constant", shape=(1,), weights=None)
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_constant")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have shape and weights parameters
        self.assertIn("shape", sig.ctor_param_names)
        self.assertIn("weights", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_activation(self):
        """Test end-to-end public API with real TensorRT add_activation."""
        spec = builtin("add_activation", type=0)
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_activation")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input and type parameters
        self.assertIn("input", sig.ctor_param_names)
        self.assertIn("type", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_convolution_nd(self):
        """Test end-to-end public API with real TensorRT add_convolution_nd."""
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_convolution_nd")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have key parameters
        self.assertIn("input", sig.ctor_param_names)
        self.assertIn("num_output_maps", sig.ctor_param_names)
        self.assertIn("kernel_shape", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_pooling_nd(self):
        """Test end-to-end public API with real TensorRT add_pooling_nd."""
        spec = builtin("add_pooling_nd", type=0, window_size=(2, 2))
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_pooling_nd")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input, type, window_size parameters
        self.assertIn("input", sig.ctor_param_names)
        self.assertIn("type", sig.ctor_param_names)
        self.assertIn("window_size", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_elementwise(self):
        """Test end-to-end public API with real TensorRT add_elementwise."""
        spec = builtin("add_elementwise", op=0)
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_elementwise")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input1, input2, op parameters
        self.assertIn("input1", sig.ctor_param_names)
        self.assertIn("input2", sig.ctor_param_names)
        self.assertIn("op", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_reduce(self):
        """Test end-to-end public API with real TensorRT add_reduce."""
        spec = builtin("add_reduce", op=0, axes=1, keep_dims=True)
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_reduce")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input, op, axes parameters
        self.assertIn("input", sig.ctor_param_names)
        self.assertIn("op", sig.ctor_param_names)
        self.assertIn("axes", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_shuffle(self):
        """Test end-to-end public API with real TensorRT add_shuffle."""
        spec = builtin("add_shuffle")
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_shuffle")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input parameter
        self.assertIn("input", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_softmax(self):
        """Test end-to-end public API with real TensorRT add_softmax."""
        spec = builtin("add_softmax")
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_softmax")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input parameter
        self.assertIn("input", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_matrix_multiply(self):
        """Test end-to-end public API with real TensorRT add_matrix_multiply."""
        spec = builtin("add_matrix_multiply", op0=0, op1=0)
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_matrix_multiply")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input0, input1, op0, op1 parameters
        self.assertIn("input0", sig.ctor_param_names)
        self.assertIn("input1", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_add_scale(self):
        """Test end-to-end public API with real TensorRT add_scale."""
        spec = builtin("add_scale", mode=0, shift=None, scale=None, power=None)
        sig = get_signature_model(spec)

        # Verify SignatureModel from real TRT API
        self.assertEqual(sig.add_name, "add_scale")
        self.assertIsInstance(sig.ctor_param_names, tuple)
        # Should have input, mode parameters
        self.assertIn("input", sig.ctor_param_names)
        self.assertIn("mode", sig.ctor_param_names)
        # Should filter out 'self'
        self.assertNotIn("self", sig.ctor_param_names)

    def test_e2e_public_api_caching(self):
        """Test that public API properly caches SignatureModel."""
        spec1 = builtin("add_constant", shape=(1,), weights=None)
        sig1 = get_signature_model(spec1)

        spec2 = builtin("add_constant", shape=(1,), weights=None)
        sig2 = get_signature_model(spec2)

        # Should return the same cached instance
        self.assertIs(sig1, sig2)

    def test_parse_param_names_malformed_no_open_paren(self):
        """Test parsing when line found but no opening parenthesis after add_name."""
        doc = "add_constant something without open paren"
        result = _parse_param_names_from_doc(doc, "add_constant")
        self.assertEqual(result, tuple())

    def test_parse_param_names_malformed_no_close_paren_after_open(self):
        """Test parsing when opening paren found but no closing."""
        doc = "add_constant(self, shape: tensorrt.Dims"
        result = _parse_param_names_from_doc(doc, "add_constant")
        self.assertEqual(result, tuple())

    def test_parse_param_names_whitespace_only_param(self):
        """Test parsing with whitespace-only parameter names."""
        doc = "add_constant(self, : tensorrt.Dims) -> IConstantLayer"
        result = _parse_param_names_from_doc(doc, "add_constant")
        # Should handle gracefully and skip empty names
        self.assertEqual(result, tuple())

    def test_signature_model_required_params(self):
        """SignatureModel.required_params contains all non-defaulted, non-ITensor params."""
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        sig = get_signature_model(spec)

        self.assertIsNotNone(sig.required_params)
        # num_output_maps, kernel_shape, and kernel are all required (no default values).
        # The input ITensor is not a required_param — it is bound separately as a TRT tensor.
        self.assertIn("num_output_maps", sig.required_params)
        self.assertIn("kernel_shape", sig.required_params)
        self.assertIn("kernel", sig.required_params)

    def test_signature_model_tensor_params(self):
        """Test that SignatureModel correctly identifies tensor input params."""
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        sig = get_signature_model(spec)

        # Verify tensor_params exists and is populated
        self.assertIsNotNone(sig.tensor_params)
        # input should be in tensor_params (it's ITensor)
        self.assertIn("input", sig.tensor_params)
        # num_output_maps should NOT be in tensor_params (it's int)
        self.assertNotIn("num_output_maps", sig.tensor_params)

    def test_signature_model_tensor_not_in_required(self):
        """Test that ITensor params are not in required_params."""
        spec = builtin("add_activation", type=0)
        sig = get_signature_model(spec)

        # input is ITensor, should be in tensor_params
        self.assertIn("input", sig.tensor_params)
        # input should NOT be in required_params
        self.assertNotIn("input", sig.required_params)
        # type is required config, should be in required_params
        self.assertIn("type", sig.required_params)

    def test_signature_model_optional_params_not_required(self):
        """Test that params with defaults are not in required_params."""
        # add_convolution_nd has bias with default = None
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        sig = get_signature_model(spec)

        # bias has default, should NOT be in required_params
        if "bias" in sig.ctor_param_names:
            self.assertNotIn("bias", sig.required_params)

    def test_signature_model_elementwise_multiple_tensors(self):
        """Test SignatureModel with method that has multiple tensor inputs."""
        spec = builtin("add_elementwise", op=0)
        sig = get_signature_model(spec)

        # add_elementwise has input1, input2 (both ITensor)
        self.assertIn("input1", sig.tensor_params)
        self.assertIn("input2", sig.tensor_params)
        # input1, input2 should NOT be in required_params
        self.assertNotIn("input1", sig.required_params)
        self.assertNotIn("input2", sig.required_params)
        # op is required config
        self.assertIn("op", sig.required_params)

    def test_weights_vs_itensor_distinction(self):
        """Test that Weights parameters are distinguished from ITensor parameters."""
        # add_convolution_nd has both ITensor (input) and Weights (kernel, bias)
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        sig = get_signature_model(spec)

        # input is ITensor - should be in tensor_params, not required_params
        self.assertIn("input", sig.tensor_params)
        self.assertNotIn("input", sig.required_params)

        # kernel is Weights - should be in required_params, NOT in tensor_params
        self.assertIn("kernel", sig.required_params)
        self.assertNotIn("kernel", sig.tensor_params)

        # num_output_maps is int - should be in required_params, not tensor_params
        self.assertIn("num_output_maps", sig.required_params)
        self.assertNotIn("num_output_maps", sig.tensor_params)

    def test_scale_layer_weights_identification(self):
        """Test that add_scale correctly identifies weights parameters."""
        spec = builtin("add_scale", mode=0, shift=None, scale=None, power=None)
        sig = get_signature_model(spec)

        # input is ITensor - runtime input
        self.assertIn("input", sig.tensor_params)
        self.assertNotIn("input", sig.required_params)

        # mode is required config parameter
        self.assertIn("mode", sig.required_params)

        # shift, scale, power are Weights - check they're not in tensor_params
        for param in ["shift", "scale", "power"]:
            if param in sig.ctor_param_names:
                self.assertNotIn(param, sig.tensor_params)

    def test_only_itensor_ilayer_are_tensor_params(self):
        """Test that only ITensor and ILayer types are classified as tensor_params."""
        # Test various methods to ensure Weights, Dims, enums are not tensor_params
        test_cases = [
            ("add_convolution_nd", {"num_output_maps": 64, "kernel_shape": (3, 3), "kernel": None}),
            ("add_pooling_nd", {"type": 0, "window_size": (2, 2)}),
            ("add_constant", {"shape": (1, 2, 3), "weights": None}),
        ]

        for add_name, kwargs in test_cases:
            with self.subTest(method=add_name):
                spec = builtin(add_name, **kwargs)
                sig = get_signature_model(spec)

                # Verify tensor_params only contains params with ITensor/ILayer in type
                for param in sig.tensor_params:
                    self.assertIn("input", param.lower(),
                                  f"{add_name}: {param} in tensor_params but doesn't look like ITensor")

    def test_all_param_categories_mutually_exclusive(self):
        """Test that tensor_params and required_params are mutually exclusive."""
        test_methods = [
            ("add_convolution_nd", {"num_output_maps": 64, "kernel_shape": (3, 3), "kernel": None}),
            ("add_activation", {"type": 0}),
            ("add_elementwise", {"op": 0}),
            ("add_pooling_nd", {"type": 0, "window_size": (2, 2)}),
        ]

        for add_name, kwargs in test_methods:
            with self.subTest(method=add_name):
                spec = builtin(add_name, **kwargs)
                sig = get_signature_model(spec)

                # tensor_params and required_params should have no overlap
                overlap = sig.tensor_params & sig.required_params
                self.assertEqual(len(overlap), 0,
                                f"{add_name}: params in both tensor_params and required_params: {overlap}")

    def test_method_with_many_parameters(self):
        """Test parsing method with many parameters of different types."""
        # add_convolution_nd has: input (ITensor), num_output_maps (int),
        # kernel_shape (Dims), kernel (Weights), bias (Weights, optional)
        spec = builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)
        sig = get_signature_model(spec)

        # Should have all parameters
        self.assertGreaterEqual(len(sig.ctor_param_names), 4)

        # Should have exactly 1 tensor param (input)
        self.assertEqual(len(sig.tensor_params), 1)
        self.assertEqual(sig.tensor_params, {"input"})

        # Should have multiple required params (excluding input and optional bias)
        self.assertGreaterEqual(len(sig.required_params), 3)
        self.assertIn("num_output_maps", sig.required_params)
        self.assertIn("kernel_shape", sig.required_params)
        self.assertIn("kernel", sig.required_params)


if __name__ == "__main__":
    unittest.main()
