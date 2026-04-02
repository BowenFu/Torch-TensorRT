"""Unit tests for tta kernel specs and CustomPluginSpec.

Covers:
  - TritonSpec, CuTileSpec, CuTeDSLSpec: creation, field validation, cache keys
  - CustomPluginSpec: construction from specs, meta_impl, op_name stability,
    multi-variant support, immutability
  - normalize_impl_to_spec: coerces bare specs to CustomPluginSpec
"""

import unittest

import torch_tensorrt.annotation as tta


# ===========================================================================
# TritonSpec
# ===========================================================================

class TestTritonSpec(unittest.TestCase):

    def test_basic_creation(self):
        def kernel(x, out):
            pass

        spec = tta.triton(kernel)
        self.assertIsInstance(spec, tta.TritonSpec)
        self.assertIs(spec.launch_fn, kernel)
        self.assertIsNone(spec.configs)

    def test_with_configs(self):
        def kernel(x, out, BLOCK: int):
            pass

        configs = [{"BLOCK": 64}, {"BLOCK": 128}]
        spec = tta.triton(kernel, configs=configs)
        self.assertEqual(spec.configs, configs)

    def test_with_tensor_formats(self):
        def kernel(x, out):
            pass

        spec = tta.triton(kernel, input_formats=["NCHW"], output_formats=["NCHW"])
        self.assertEqual(spec.input_formats, ["NCHW"])
        self.assertEqual(spec.output_formats, ["NCHW"])

    def test_with_kwargs(self):
        def kernel(x, out):
            pass

        spec = tta.triton(kernel, num_warps=4)
        self.assertEqual(spec.kwargs["num_warps"], 4)

    def test_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            tta.TritonSpec(launch_fn="not_callable")

    def test_rejects_non_list_configs(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError):
            tta.TritonSpec(launch_fn=kernel, configs={"BLOCK": 128})

    def test_cache_key_is_stable(self):
        def kernel(x, out):
            pass

        spec = tta.triton(kernel, configs=[{"BLOCK": 128}])
        self.assertEqual(spec.to_cache_key(), spec.to_cache_key())

    def test_cache_key_backend_tag(self):
        def kernel(x, out):
            pass

        key = tta.triton(kernel).to_cache_key()
        self.assertEqual(key[0], "triton")


# ===========================================================================
# CuTileSpec
# ===========================================================================

class TestCuTileSpec(unittest.TestCase):

    def test_basic_creation(self):
        def kernel(x, out):
            pass

        spec = tta.cutile(kernel)
        self.assertIsInstance(spec, tta.CuTileSpec)
        self.assertIs(spec.launch_fn, kernel)

    def test_with_configs(self):
        def kernel(x, out):
            pass

        configs = [{"TILE_M": 64, "TILE_N": 64}]
        spec = tta.cutile(kernel, configs=configs)
        self.assertEqual(spec.configs, configs)

    def test_cache_key_backend_tag(self):
        def kernel(x, out):
            pass

        key = tta.cutile(kernel).to_cache_key()
        self.assertEqual(key[0], "cutile")


# ===========================================================================
# CuTeDSLSpec
# ===========================================================================

class TestCuTeDSLSpec(unittest.TestCase):

    def test_basic_creation(self):
        def kernel(x, out):
            pass

        spec = tta.cutedsl(kernel)
        self.assertIsInstance(spec, tta.CuTeDSLSpec)
        self.assertIs(spec.launch_fn, kernel)
        self.assertIsNone(spec.arch)

    def test_with_arch(self):
        def kernel(x, out):
            pass

        spec = tta.cutedsl(kernel, arch="sm_90")
        self.assertEqual(spec.arch, "sm_90")

    def test_with_configs_and_arch(self):
        def kernel(x, out):
            pass

        spec = tta.cutedsl(kernel, arch="sm_80", configs=[{"BLOCK": 128}])
        self.assertEqual(spec.arch, "sm_80")
        self.assertEqual(spec.configs, [{"BLOCK": 128}])

    def test_cache_key_backend_tag(self):
        def kernel(x, out):
            pass

        key = tta.cutedsl(kernel).to_cache_key()
        self.assertEqual(key[0], "cutedsl")

    def test_cache_key_includes_arch(self):
        def kernel(x, out):
            pass

        key = tta.cutedsl(kernel, arch="sm_90").to_cache_key()
        self.assertEqual(key[2], "sm_90")


# ===========================================================================
# CustomPluginSpec
# ===========================================================================

class TestCustomPluginSpec(unittest.TestCase):

    def test_single_triton_spec(self):
        def kernel(x, out):
            pass

        meta = lambda x: x.new_empty(x.shape)
        spec = tta.triton(kernel, configs=[{"BLOCK": 128}])
        descriptor = tta.custom_plugin(spec, meta_impl=meta)
        self.assertIsInstance(descriptor, tta.CustomPluginSpec)
        self.assertEqual(len(descriptor.specs), 1)
        self.assertIs(descriptor.specs[0], spec)
        self.assertIs(descriptor.meta_impl, meta)

    def test_single_cutile_spec(self):
        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.cutile(kernel), meta_impl=lambda x: x.new_empty(x.shape))
        self.assertIsInstance(descriptor.specs[0], tta.CuTileSpec)

    def test_single_cutedsl_spec(self):
        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.cutedsl(kernel, arch="sm_90"), meta_impl=lambda x: x.new_empty(x.shape))
        self.assertIsInstance(descriptor.specs[0], tta.CuTeDSLSpec)

    def test_multiple_variants(self):
        def k1(x, out):
            pass

        def k2(x, out):
            pass

        descriptor = tta.custom_plugin(
            [tta.triton(k1, configs=[{"BLOCK": 64}]), tta.triton(k2, configs=[{"BLOCK": 128}])],
            meta_impl=lambda x: x.new_empty(x.shape),
        )
        self.assertEqual(len(descriptor.specs), 2)

    def test_with_meta_impl(self):
        def kernel(x, out):
            pass

        meta = lambda x: x.new_empty(x.shape)
        descriptor = tta.custom_plugin(tta.triton(kernel), meta_impl=meta)
        self.assertIs(descriptor.meta_impl, meta)

    def test_rejects_missing_meta_impl(self):
        def kernel(x, out):
            pass

        with self.assertRaises((TypeError, ValueError)):
            tta.custom_plugin(tta.triton(kernel))

    def test_rejects_none_meta_impl(self):
        def kernel(x, out):
            pass

        with self.assertRaises(ValueError) as ctx:
            tta.custom_plugin(tta.triton(kernel), meta_impl=None)
        self.assertIn("meta_impl", str(ctx.exception))

    def test_rejects_invalid_type(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin("not_a_spec", meta_impl=lambda x: x.new_empty(x.shape))
        msg = str(ctx.exception)
        self.assertIn("TritonSpec", msg)
        self.assertIn("CuTileSpec", msg)
        self.assertIn("CuTeDSLSpec", msg)

    def test_rejects_invalid_element_in_list(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError):
            tta.custom_plugin([tta.triton(kernel), "invalid"], meta_impl=lambda x: x.new_empty(x.shape))

    def test_rejects_empty_list(self):
        with self.assertRaises(ValueError) as ctx:
            tta.custom_plugin([], meta_impl=lambda x: x.new_empty(x.shape))
        self.assertIn("empty", str(ctx.exception))

    def test_rejects_non_callable_meta_impl(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin(tta.triton(kernel), meta_impl="not_callable")
        self.assertIn("callable", str(ctx.exception))

    def test_op_name_format(self):
        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel), meta_impl=lambda x: x.new_empty(x.shape))
        self.assertIn("::", descriptor.op_name)
        ns, name = descriptor.op_name.split("::", 1)
        self.assertEqual(ns, "tta_custom")
        self.assertTrue(len(name) > 0)

    def test_op_name_is_deterministic(self):
        def kernel(x, out):
            pass

        meta = lambda x: x.new_empty(x.shape)
        spec = tta.triton(kernel)
        self.assertEqual(tta.custom_plugin(spec, meta_impl=meta).op_name,
                         tta.custom_plugin(spec, meta_impl=meta).op_name)

    def test_op_name_differs_across_kernel_functions(self):
        def kernel_a(x, out):
            pass

        def kernel_b(x, out):
            pass

        meta = lambda x: x.new_empty(x.shape)
        op_name_a = tta.custom_plugin(tta.triton(kernel_a), meta_impl=meta).op_name
        op_name_b = tta.custom_plugin(tta.triton(kernel_b), meta_impl=meta).op_name
        self.assertNotEqual(op_name_a, op_name_b)

    def test_is_immutable(self):
        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel), meta_impl=lambda x: x.new_empty(x.shape))
        with self.assertRaises((AttributeError, TypeError)):
            descriptor.op_name = "hacked"


# ===========================================================================
# normalize_impl_to_spec
# ===========================================================================

class TestNormalizeImplToSpec(unittest.TestCase):

    def test_bare_triton_spec_rejected(self):
        def kernel(x, out):
            pass

        spec = tta.triton(kernel)
        with self.assertRaises(TypeError) as ctx:
            tta.normalize_impl_to_spec(spec)
        self.assertIn("meta_impl", str(ctx.exception))

    def test_bare_cutile_spec_rejected(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError) as ctx:
            tta.normalize_impl_to_spec(tta.cutile(kernel))
        self.assertIn("meta_impl", str(ctx.exception))

    def test_bare_cutedsl_spec_rejected(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError) as ctx:
            tta.normalize_impl_to_spec(tta.cutedsl(kernel))
        self.assertIn("meta_impl", str(ctx.exception))

    def test_custom_plugin_spec_passthrough(self):
        def kernel(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel), meta_impl=lambda x: x.new_empty(x.shape))
        self.assertIs(tta.normalize_impl_to_spec(descriptor), descriptor)

    def test_rejects_invalid_type(self):
        with self.assertRaisesRegex(TypeError, "normalize_impl_to_spec"):
            tta.normalize_impl_to_spec(123)


# ===========================================================================
# AnnotationMetadata helpers
# ===========================================================================

class TestAnnotationMetadata(unittest.TestCase):
    def test_attach_and_get_round_trip(self):
        from torch_tensorrt.annotation._specs import (
            AnnotationMetadata,
            attach_annotation_metadata,
            get_annotation_metadata,
        )

        def fn():
            pass

        meta = AnnotationMetadata(impl=None)
        attach_annotation_metadata(fn, meta)
        retrieved = get_annotation_metadata(fn)
        self.assertIs(retrieved, meta)

    def test_get_returns_none_when_absent(self):
        from torch_tensorrt.annotation._specs import get_annotation_metadata

        def fn():
            pass

        self.assertIsNone(get_annotation_metadata(fn))


# ===========================================================================
# KernelImplSpec
# ===========================================================================

class TestKernelImplSpec(unittest.TestCase):
    def test_single_spec_cache_key(self):
        def kernel(x, out):
            pass

        spec = tta.triton(kernel)
        ki = tta.KernelImplSpec(kernel=spec)
        key = ki.to_cache_key()
        self.assertEqual(key[0], "custom_plugin")

    def test_list_kernel_cache_key(self):
        def k1(x, out):
            pass

        def k2(x, out):
            pass

        ki = tta.KernelImplSpec(kernel=[tta.triton(k1), tta.triton(k2)])
        key = ki.to_cache_key()
        self.assertEqual(key[0], "custom_plugin")
        self.assertIsInstance(key[1], tuple)
        self.assertEqual(len(key[1]), 2)

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            tta.KernelImplSpec(kernel=[])

    def test_invalid_item_in_list_raises(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError):
            tta.KernelImplSpec(kernel=[tta.triton(kernel), "bad"])

    def test_invalid_kernel_type_raises(self):
        with self.assertRaises(TypeError):
            tta.KernelImplSpec(kernel="not_a_spec")

    def test_non_callable_meta_impl_raises(self):
        def kernel(x, out):
            pass

        with self.assertRaises(TypeError):
            tta.KernelImplSpec(kernel=tta.triton(kernel), meta_impl="not_callable")


if __name__ == "__main__":
    unittest.main()
