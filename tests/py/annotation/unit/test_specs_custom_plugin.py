"""Spec creation and validation for custom_plugin backends: Triton, CuTile, CuTeDSL, CustomPluginSpec."""

import unittest

import torch_tensorrt.annotation as tta


class TestTritonSpec(unittest.TestCase):
    """Tests for TritonSpec creation and validation."""

    def test_triton_spec_basic(self):
        """Test basic TritonSpec creation."""

        def my_kernel(x, y, out, BLOCK: int):
            pass

        spec = tta.triton(my_kernel)

        self.assertIsInstance(spec, tta.TritonSpec)
        self.assertEqual(spec.launch_fn, my_kernel)
        self.assertIsNone(spec.configs)
        self.assertIsNone(spec.input_formats)
        self.assertIsNone(spec.output_formats)

    def test_triton_spec_with_configs(self):
        """Test TritonSpec with multiple configs."""

        def kernel(x, out, BLOCK: int, TILE_M: int):
            pass

        configs = [{"BLOCK": 64, "TILE_M": 32}, {"BLOCK": 128, "TILE_M": 64}]

        spec = tta.triton(kernel, configs=configs)

        self.assertEqual(spec.configs, configs)
        self.assertEqual(len(spec.configs), 2)

    def test_triton_spec_with_formats(self):
        """Test TritonSpec with tensor format hints."""

        def kernel(x, out):
            pass

        spec = tta.triton(
            kernel, input_formats=["NCHW"], output_formats=["NCHW"]
        )

        self.assertEqual(spec.input_formats, ["NCHW"])
        self.assertEqual(spec.output_formats, ["NCHW"])

    def test_triton_spec_with_kwargs(self):
        """Test TritonSpec with additional kwargs."""

        def kernel(x, out):
            pass

        spec = tta.triton(kernel, name="my_kernel", num_warps=4)

        self.assertEqual(spec.kwargs["name"], "my_kernel")
        self.assertEqual(spec.kwargs["num_warps"], 4)

    def test_triton_spec_invalid_launch_fn(self):
        """Test TritonSpec rejects non-callable launch_fn."""

        with self.assertRaises(TypeError) as ctx:
            tta.TritonSpec(launch_fn="not_a_function")

        self.assertIn("callable", str(ctx.exception))

    def test_triton_spec_invalid_configs(self):
        """Test TritonSpec rejects non-list configs."""

        def kernel(x, out):
            pass

        with self.assertRaises(TypeError) as ctx:
            tta.TritonSpec(launch_fn=kernel, configs="not_a_list")

        self.assertIn("list", str(ctx.exception))

    def test_triton_spec_cache_key(self):
        """Test TritonSpec generates cache key."""

        def kernel(x, out, BLOCK: int):
            pass

        spec = tta.triton(kernel, configs=[{"BLOCK": 128}])

        key = spec.to_cache_key()

        self.assertIsInstance(key, tuple)
        self.assertEqual(key[0], "triton")
        self.assertIsInstance(key[1], int)  # id(launch_fn)


class TestCuTileSpec(unittest.TestCase):
    """Tests for CuTileSpec creation and validation."""

    def test_cutile_spec_basic(self):
        """Test basic CuTileSpec creation."""

        def my_kernel(x, y, out, TILE_M: int):
            pass

        spec = tta.cutile(my_kernel)

        self.assertIsInstance(spec, tta.CuTileSpec)
        self.assertEqual(spec.launch_fn, my_kernel)

    def test_cutile_spec_with_configs(self):
        """Test CuTileSpec with configs."""

        def kernel(x, out):
            pass

        configs = [{"TILE_M": 64, "TILE_N": 64}]
        spec = tta.cutile(kernel, configs=configs)

        self.assertEqual(spec.configs, configs)

    def test_cutile_spec_cache_key(self):
        """Test CuTileSpec generates cache key."""

        def kernel(x, out):
            pass

        spec = tta.cutile(kernel)
        key = spec.to_cache_key()

        self.assertEqual(key[0], "cutile")


class TestCuTeDSLSpec(unittest.TestCase):
    """Tests for CuTeDSLSpec creation and validation."""

    def test_cutedsl_spec_basic(self):
        """Test basic CuTeDSLSpec creation."""

        def my_kernel(x, y, out):
            pass

        spec = tta.cutedsl(my_kernel)

        self.assertIsInstance(spec, tta.CuTeDSLSpec)
        self.assertEqual(spec.launch_fn, my_kernel)
        self.assertIsNone(spec.arch)

    def test_cutedsl_spec_with_arch(self):
        """Test CuTeDSLSpec with architecture target."""

        def kernel(x, out):
            pass

        spec = tta.cutedsl(kernel, arch="sm_90")

        self.assertEqual(spec.arch, "sm_90")

    def test_cutedsl_spec_with_configs_and_arch(self):
        """Test CuTeDSLSpec with both configs and arch."""

        def kernel(x, out):
            pass

        spec = tta.cutedsl(
            kernel, arch="sm_80", configs=[{"BLOCK": 128}]
        )

        self.assertEqual(spec.arch, "sm_80")
        self.assertEqual(spec.configs, [{"BLOCK": 128}])

    def test_cutedsl_spec_cache_key(self):
        """Test CuTeDSLSpec cache key includes arch."""

        def kernel(x, out):
            pass

        spec = tta.cutedsl(kernel, arch="sm_90")
        key = spec.to_cache_key()

        self.assertEqual(key[0], "cutedsl")
        self.assertEqual(key[2], "sm_90")  # arch in key


class TestCustomPluginSpec(unittest.TestCase):
    """Tests for CustomPluginSpec returned by custom_plugin()."""

    def test_custom_plugin_single_triton(self):
        """Test custom_plugin with single TritonSpec returns CustomPluginSpec."""

        def kernel(x, out, BLOCK: int):
            pass

        triton_spec = tta.triton(kernel, configs=[{"BLOCK": 128}])
        descriptor = tta.custom_plugin(triton_spec)

        self.assertIsInstance(descriptor, tta.CustomPluginSpec)
        self.assertEqual(len(descriptor.specs), 1)
        self.assertIs(descriptor.specs[0], triton_spec)
        self.assertIsNone(descriptor.meta_impl)

    def test_custom_plugin_single_cutile(self):
        """Test custom_plugin with single CuTileSpec."""

        def kernel(x, out):
            pass

        cutile_spec = tta.cutile(kernel)
        descriptor = tta.custom_plugin(cutile_spec)

        self.assertIsInstance(descriptor, tta.CustomPluginSpec)
        self.assertIsInstance(descriptor.specs[0], tta.CuTileSpec)

    def test_custom_plugin_single_cutedsl(self):
        """Test custom_plugin with single CuTeDSLSpec."""

        def kernel(x, out):
            pass

        cutedsl_spec = tta.cutedsl(kernel, arch="sm_90")
        descriptor = tta.custom_plugin(cutedsl_spec)

        self.assertIsInstance(descriptor, tta.CustomPluginSpec)
        self.assertIsInstance(descriptor.specs[0], tta.CuTeDSLSpec)

    def test_custom_plugin_multiple_variants(self):
        """Test custom_plugin with list of kernel specs."""

        def kernel1(x, out, BLOCK: int):
            pass

        def kernel2(x, out, BLOCK: int):
            pass

        triton1 = tta.triton(kernel1, configs=[{"BLOCK": 64}])
        triton2 = tta.triton(kernel2, configs=[{"BLOCK": 128}])

        descriptor = tta.custom_plugin([triton1, triton2])

        self.assertIsInstance(descriptor, tta.CustomPluginSpec)
        self.assertEqual(len(descriptor.specs), 2)

    def test_custom_plugin_with_meta_impl(self):
        """Test custom_plugin with metadata callable."""

        def kernel(x, out):
            pass

        def meta_fn(x):
            return {"shape": x.shape, "dtype": x.dtype}

        triton_spec = tta.triton(kernel)
        descriptor = tta.custom_plugin(triton_spec, meta_impl=meta_fn)

        self.assertEqual(descriptor.meta_impl, meta_fn)

    def test_custom_plugin_invalid_kernel_type(self):
        """Test custom_plugin rejects invalid kernel type."""

        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin("not_a_kernel_spec")

        self.assertIn("TritonSpec", str(ctx.exception))
        self.assertIn("CuTileSpec", str(ctx.exception))
        self.assertIn("CuTeDSLSpec", str(ctx.exception))

    def test_custom_plugin_invalid_kernel_list(self):
        """Test custom_plugin rejects list with invalid types."""

        def kernel(x, out):
            pass

        triton_spec = tta.triton(kernel)

        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin([triton_spec, "invalid"])

        # Error mentions valid types
        self.assertIn("CuTileSpec", str(ctx.exception))

    def test_custom_plugin_empty_list(self):
        """Test custom_plugin rejects empty kernel list."""

        with self.assertRaises(ValueError) as ctx:
            tta.custom_plugin([])

        self.assertIn("empty", str(ctx.exception))

    def test_custom_plugin_invalid_meta_impl(self):
        """Test custom_plugin rejects non-callable meta_impl."""

        def kernel(x, out):
            pass

        triton_spec = tta.triton(kernel)

        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin(triton_spec, meta_impl="not_callable")

        self.assertIn("callable", str(ctx.exception))

    def test_custom_plugin_op_name_format(self):
        """CustomPluginSpec.op_name has 'ns::name' format."""

        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel))

        self.assertIn("::", descriptor.op_name)
        ns, name = descriptor.op_name.split("::", 1)
        self.assertEqual(ns, "tta_custom")
        self.assertTrue(len(name) > 0)

    def test_custom_plugin_op_name_is_deterministic(self):
        """Same specs always produce the same op_name."""

        def kernel(x, out):
            pass

        spec = tta.triton(kernel)
        d1 = tta.custom_plugin(spec)
        d2 = tta.custom_plugin(spec)

        self.assertEqual(d1.op_name, d2.op_name)

    def test_custom_plugin_descriptor_is_frozen(self):
        """CustomPluginSpec is immutable."""

        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel))

        with self.assertRaises((AttributeError, TypeError)):
            descriptor.op_name = "hacked"


if __name__ == "__main__":
    unittest.main()
