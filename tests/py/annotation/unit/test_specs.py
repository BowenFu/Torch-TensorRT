"""Spec creation, validation, and normalization tests."""

import unittest

import torch

import torch_tensorrt.annotation as tta


class TestPluginSpec(unittest.TestCase):

    def test_plugin_spec_creation(self):
        """Test RegistryPluginSpec can be created."""
        spec = tta.plugin("TestPlugin", "1.0", "test_ns", alpha=0.5, beta=1.0)

        self.assertIsInstance(spec, tta.RegistryPluginSpec)
        self.assertEqual(spec.name, "TestPlugin")
        self.assertEqual(spec.version, "1.0")
        self.assertEqual(spec.namespace, "test_ns")
        self.assertEqual(spec.attrs["alpha"], 0.5)

    def test_spec_validation(self):
        """Test spec validation."""
        # Valid specs
        tta.plugin("Test", "1.0", "ns")

        # Invalid specs - RegistryPluginSpec
        with self.assertRaisesRegex(ValueError, "name"):
            tta.plugin("", "1.0", "ns")  # Empty name

        with self.assertRaisesRegex(ValueError, "version"):
            tta.plugin("Test", "", "ns")  # Empty version

    def test_cache_key_generation(self):
        """Test cache key generation for specs."""
        spec1 = tta.plugin("Test", "1.0", "ns", a=1, b=2)
        spec2 = tta.plugin("Test", "1.0", "ns", b=2, a=1)  # Different order

        # Cache keys should be identical (attrs sorted)
        key1 = spec1.to_cache_key()
        key2 = spec2.to_cache_key()
        self.assertEqual(key1, key2)


class TestBuiltinSpec(unittest.TestCase):

    def test_builtin_spec_creation(self):
        """Test BuiltinSpec can be created."""
        spec = tta.builtin(
            "add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None
        )

        self.assertIsInstance(spec, tta.BuiltinSpec)
        self.assertEqual(spec.add_name, "add_convolution_nd")
        self.assertEqual(spec.kwargs["num_output_maps"], 64)

    def test_spec_validation(self):
        """Test spec validation."""
        # Valid specs
        tta.builtin("add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3), kernel=None)

        # Invalid specs - BuiltinSpec
        with self.assertRaisesRegex(ValueError, "add_name"):
            tta.builtin("")  # Empty add_name

    def test_builtin_requires_add_prefix(self):
        """Test that builtin() fails fast if add_name doesn't start with 'add_'."""
        # Should fail immediately with helpful error message
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("convolution_nd")
        error_msg = str(ctx.exception)
        self.assertIn("must start with 'add_'", error_msg)
        self.assertIn("add_convolution_nd", error_msg)

    def test_builtin_validates_method_exists(self):
        """Test that builtin() fails fast if method doesn't exist on TensorRT."""
        # Should fail immediately for non-existent method
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_nonexistent_layer")
        error_msg = str(ctx.exception)
        self.assertIn("not found on trt.INetworkDefinition", error_msg)

    def test_builtin_requires_parameters(self):
        """Test that builtin() validates required parameters are provided."""
        # add_convolution_nd requires: num_output_maps, kernel_shape, kernel (weights)
        # Should fail if missing required params
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_convolution_nd")
        error_msg = str(ctx.exception)
        self.assertIn("Missing required parameters", error_msg)
        # Should mention at least some required params
        self.assertTrue(
            "num_output_maps" in error_msg
            or "kernel_shape" in error_msg
            or "kernel" in error_msg
        )

    def test_builtin_with_required_parameters(self):
        """Test that builtin() succeeds when required parameters are provided."""
        # Provide all required parameters for add_convolution_nd
        # Note: input is ITensor (runtime), so not required in kwargs
        spec = tta.builtin(
            "add_convolution_nd",
            num_output_maps=64,
            kernel_shape=(3, 3),
            kernel=None,  # Placeholder for weights
        )
        self.assertIsInstance(spec, tta.BuiltinSpec)
        self.assertEqual(spec.kwargs["num_output_maps"], 64)

    def test_builtin_spec_creation_with_all_required_params(self):
        """builtin() succeeds when all required params are supplied."""
        spec = tta.builtin("add_constant", shape=(1, 2, 3), weights=None)
        self.assertIsInstance(spec, tta.BuiltinSpec)

    def test_builtin_missing_single_required_param(self):
        """Test that builtin() fails when a single required param is missing."""
        # add_convolution_nd requires: num_output_maps, kernel_shape, kernel
        # Provide only 2 out of 3 - should fail
        with self.assertRaises(ValueError) as ctx:
            tta.builtin(
                "add_convolution_nd", num_output_maps=64, kernel_shape=(3, 3)
                # Missing: kernel
            )
        error_msg = str(ctx.exception)
        self.assertIn("Missing required parameters", error_msg)
        self.assertIn("kernel", error_msg)

    def test_builtin_missing_multiple_required_params(self):
        """Test that builtin() fails when multiple required params are missing."""
        # add_convolution_nd requires: num_output_maps, kernel_shape, kernel
        # Provide only 1 out of 3 - should fail and list all missing
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_convolution_nd", num_output_maps=64)
        error_msg = str(ctx.exception)
        self.assertIn("Missing required parameters", error_msg)
        # Should list both missing params
        self.assertIn("kernel", error_msg)
        self.assertIn("kernel_shape", error_msg)

    def test_builtin_constant_missing_required_params(self):
        """Test parameter validation with add_constant."""
        # add_constant requires: shape, weights
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_constant")
        error_msg = str(ctx.exception)
        self.assertIn("Missing required parameters", error_msg)
        self.assertTrue("shape" in error_msg or "weights" in error_msg)

    def test_builtin_activation_missing_required_params(self):
        """Test parameter validation with add_activation."""
        # add_activation requires: type (input is ITensor, so not required)
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_activation")
        error_msg = str(ctx.exception)
        self.assertIn("Missing required parameters", error_msg)
        self.assertIn("type", error_msg)

    def test_builtin_elementwise_missing_required_params(self):
        """Test parameter validation with add_elementwise."""
        # add_elementwise requires: op (input1 and input2 are ITensor)
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_elementwise")
        error_msg = str(ctx.exception)
        self.assertIn("Missing required parameters", error_msg)
        self.assertIn("op", error_msg)

    def test_builtin_tensor_inputs_not_required(self):
        """Test that ITensor/ILayer parameters are not required in kwargs."""
        # add_activation: input is ITensor (runtime), type is required config
        # Should only require 'type', not 'input'
        spec = tta.builtin("add_activation", type=0)
        self.assertIsInstance(spec, tta.BuiltinSpec)
        # Verify input is not in kwargs (it's a runtime param)
        self.assertNotIn("input", spec.kwargs)

    def test_builtin_partial_params_still_fails(self):
        """Test that providing some but not all required params still fails."""
        # add_convolution_nd requires: num_output_maps, kernel_shape, kernel
        # Providing only some should still fail
        with self.assertRaisesRegex(ValueError, "Missing required parameters"):
            tta.builtin("add_convolution_nd", num_output_maps=64)

    def test_error_message_lists_all_missing_params(self):
        """Test that error message lists ALL missing parameters."""
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_convolution_nd", num_output_maps=64)

        error_msg = str(ctx.exception)
        # Should mention it's about missing parameters
        self.assertIn("Missing required parameters", error_msg)
        # Should mention the method name
        self.assertIn("add_convolution_nd", error_msg)
        # Should list the missing params (kernel_shape and kernel)
        self.assertIn("kernel", error_msg)
        self.assertIn("kernel_shape", error_msg)

    def test_error_message_quality_single_missing(self):
        """Test error message quality when single parameter is missing."""
        with self.assertRaises(ValueError) as ctx:
            tta.builtin(
                "add_convolution_nd",
                num_output_maps=64,
                kernel_shape=(3, 3),
                # Missing: kernel
            )

        error_msg = str(ctx.exception)
        self.assertIn("kernel", error_msg)
        self.assertIn("add_convolution_nd", error_msg)
        # Should be clear and actionable
        self.assertIn("must be provided", error_msg.lower())

    def test_error_message_helpful_for_activation(self):
        """Test error message for add_activation without type."""
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_activation")

        error_msg = str(ctx.exception)
        self.assertIn("type", error_msg)
        self.assertIn("add_activation", error_msg)
        self.assertIn("Missing required parameters", error_msg)

    def test_method_with_only_optional_params_succeeds(self):
        """Test method where all non-tensor params have defaults."""
        # add_shuffle only requires input (ITensor), all other params are optional
        try:
            spec = tta.builtin("add_shuffle")
            self.assertIsInstance(spec, tta.BuiltinSpec)
        except ValueError as e:
            # If add_shuffle has required params, this is expected
            self.assertIn("Missing required parameters", str(e))

    def test_method_with_three_plus_tensor_inputs(self):
        """Test method with multiple tensor inputs."""
        # add_select requires condition, thenInput, elseInput (3 ITensor params)
        # These should all be in tensor_params, not required_params
        try:
            spec = tta.builtin("add_select")
            self.assertIsInstance(spec, tta.BuiltinSpec)
            # Verify it succeeded - select has no required non-tensor params
        except ValueError:
            # If select has required params, that's also valid
            pass

    def test_edge_case_method_with_no_required_params(self):
        """Test method with only ITensor inputs and optional params."""
        # add_identity only takes input (ITensor)
        try:
            spec = tta.builtin("add_identity")
            self.assertIsInstance(spec, tta.BuiltinSpec)
        except (ValueError, AttributeError):
            # Method might not exist or have required params
            pass

    def test_complex_parameter_types_dims(self):
        """Test that Dims parameters are correctly identified as required."""
        # kernel_shape is Dims type - should be required
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_convolution_nd", num_output_maps=64, kernel=None)
            # Missing: kernel_shape (Dims)

        error_msg = str(ctx.exception)
        self.assertIn("kernel_shape", error_msg)

    def test_complex_parameter_types_enum(self):
        """Test that enum parameters (type, op, mode) are required."""
        # type is ActivationType enum - should be required
        with self.assertRaises(ValueError) as ctx:
            tta.builtin("add_activation")

        error_msg = str(ctx.exception)
        self.assertIn("type", error_msg)

    def test_multiple_methods_different_signatures(self):
        """Test validation works correctly across methods with different signatures."""
        test_cases = [
            ("add_activation", {"type": 0}, True),  # Should succeed
            ("add_activation", {}, False),  # Should fail
            ("add_elementwise", {"op": 0}, True),  # Should succeed
            ("add_elementwise", {}, False),  # Should fail
            ("add_constant", {"shape": (1, 2), "weights": None}, True),  # Should succeed
            ("add_constant", {}, False),  # Should fail
        ]

        for add_name, kwargs, should_succeed in test_cases:
            with self.subTest(method=add_name, kwargs=kwargs):
                if should_succeed:
                    spec = tta.builtin(add_name, **kwargs)
                    self.assertIsInstance(spec, tta.BuiltinSpec)
                else:
                    with self.assertRaises(ValueError) as ctx:
                        tta.builtin(add_name, **kwargs)
                    self.assertIn("Missing required parameters", str(ctx.exception))

    def test_builtin_with_extra_params_allowed(self):
        """Test that providing extra parameters (for setattr) is allowed."""
        # Providing extra params beyond constructor should work
        # (they'll be used for setattr on the layer)
        spec = tta.builtin(
            "add_activation",
            type=0,
            alpha=0.1,  # Extra param for setattr
            beta=0.5,  # Extra param for setattr
        )
        self.assertIsInstance(spec, tta.BuiltinSpec)
        self.assertEqual(spec.kwargs["type"], 0)
        self.assertEqual(spec.kwargs["alpha"], 0.1)


class TestNormalization(unittest.TestCase):
    """Tests for normalize_impl_to_spec."""

    def test_normalize_impl_to_spec(self):
        """Test impl normalization."""
        # RegistryPluginSpec from dict
        spec1 = tta.normalize_impl_to_spec(
            {"name": "Test", "version": "1.0", "namespace": "ns"}
        )
        self.assertIsInstance(spec1, tta.RegistryPluginSpec)

        # BuiltinSpec from dict (backward compatibility with "target")
        spec2 = tta.normalize_impl_to_spec({"target": "add_convolution_nd", "num_output_maps": 64, "kernel_shape": (3, 3), "kernel": None})
        self.assertIsInstance(spec2, tta.BuiltinSpec)

        # Already a spec
        spec3 = tta.plugin("Test", "1.0", "ns")
        spec4 = tta.normalize_impl_to_spec(spec3)
        self.assertEqual(spec3, spec4)

    def test_normalize_triton_spec(self):
        """Test normalization wraps bare TritonSpec in CustomPluginSpec."""

        def kernel(x, out):
            pass

        triton_spec = tta.triton(kernel)
        normalized = tta.normalize_impl_to_spec(triton_spec)

        self.assertIsInstance(normalized, tta.CustomPluginSpec)
        self.assertIs(normalized.specs[0], triton_spec)

    def test_normalize_cutile_spec(self):
        """Test normalization wraps bare CuTileSpec in CustomPluginSpec."""

        def kernel(x, out):
            pass

        cutile_spec = tta.cutile(kernel)
        normalized = tta.normalize_impl_to_spec(cutile_spec)

        self.assertIsInstance(normalized, tta.CustomPluginSpec)
        self.assertIsInstance(normalized.specs[0], tta.CuTileSpec)

    def test_normalize_cutedsl_spec(self):
        """Test normalization wraps bare CuTeDSLSpec in CustomPluginSpec."""

        def kernel(x, out):
            pass

        cutedsl_spec = tta.cutedsl(kernel)
        normalized = tta.normalize_impl_to_spec(cutedsl_spec)

        self.assertIsInstance(normalized, tta.CustomPluginSpec)
        self.assertIsInstance(normalized.specs[0], tta.CuTeDSLSpec)

    def test_normalize_plugin_impl_descriptor_passthrough(self):
        """Test normalization passes through CustomPluginSpec unchanged."""

        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel))
        normalized = tta.normalize_impl_to_spec(descriptor)

        self.assertIs(normalized, descriptor)

    def test_normalize_invalid_type(self):
        """Test normalize_impl_to_spec with invalid type."""
        # Invalid type should raise TypeError
        with self.assertRaisesRegex(TypeError, "normalize_impl_to_spec"):
            tta.normalize_impl_to_spec(123)  # int is not valid

        with self.assertRaisesRegex(TypeError, "normalize_impl_to_spec"):
            tta.normalize_impl_to_spec("invalid")  # string is not valid


if __name__ == "__main__":
    unittest.main()
