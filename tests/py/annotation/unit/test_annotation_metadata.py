"""Unit tests for AnnotationMetadata: structure, defaults, and public API surface.

AnnotationMetadata is the dataclass attached to every export_as-decorated function.
It carries the impl spec, the require flag, the optional name, and any ctor kwargs.

Tests:
  TestAnnotationMetadataStructure  — field types and defaults.
  TestAnnotationMetadataPublicAPI  — accessible from tta namespace / __all__.
  TestGetAnnotationMetadata        — get_annotation_metadata helper.
  TestSelfAttrSpec                 — self_attr() returns FromMethodSelf; path stored.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation._specs import (
    AnnotationMetadata,
    FromMethodSelf,
    get_annotation_metadata,
)


# ---------------------------------------------------------------------------
# TestAnnotationMetadataStructure
# ---------------------------------------------------------------------------

class TestAnnotationMetadataStructure(unittest.TestCase):
    """AnnotationMetadata carries impl, require, name, kwargs."""

    def _make_plugin_md(self, name=None, require=False):
        spec = tta.plugin("P", "1.0", "ns")
        return AnnotationMetadata(impl=spec, kwargs={}, name=name, require=require)

    def test_impl_stored(self):
        spec = tta.plugin("P", "1.0", "ns")
        md = AnnotationMetadata(impl=spec, kwargs={}, name=None, require=False)
        self.assertIs(md.impl, spec)

    def test_name_defaults_none(self):
        md = self._make_plugin_md()
        self.assertIsNone(md.name)

    def test_name_stored(self):
        md = self._make_plugin_md(name="my_label")
        self.assertEqual(md.name, "my_label")

    def test_require_defaults_false(self):
        md = self._make_plugin_md()
        self.assertFalse(md.require)

    def test_require_true_stored(self):
        md = self._make_plugin_md(require=True)
        self.assertTrue(md.require)

    def test_kwargs_stored(self):
        spec = tta.plugin("P", "1.0", "ns")
        md = AnnotationMetadata(impl=spec, kwargs={"extra": 42}, name=None, require=False)
        self.assertEqual(md.kwargs.get("extra"), 42)

    def test_builtin_impl_stored(self):
        spec = tta.builtin("add_activation", type=0)
        md = AnnotationMetadata(impl=spec, kwargs={}, name=None, require=False)
        self.assertIsInstance(md.impl, tta.BuiltinSpec)

    def test_custom_plugin_impl_stored(self):
        def kernel(x, out):
            pass
        desc = tta.custom_plugin(tta.triton(kernel))
        md = AnnotationMetadata(impl=desc, kwargs={}, name=None, require=False)
        self.assertIsInstance(md.impl, tta.CustomPluginSpec)


# ---------------------------------------------------------------------------
# TestAnnotationMetadataPublicAPI
# ---------------------------------------------------------------------------

class TestAnnotationMetadataPublicAPI(unittest.TestCase):
    """AnnotationMetadata and related types are accessible from tta namespace."""

    def test_annotation_metadata_in_all(self):
        self.assertIn("AnnotationMetadata", tta.__all__)

    def test_annotation_metadata_accessible(self):
        self.assertTrue(hasattr(tta, "AnnotationMetadata"))

    def test_plugin_spec_in_all(self):
        self.assertIn("RegistryPluginSpec", tta.__all__)

    def test_builtin_spec_in_all(self):
        self.assertIn("BuiltinSpec", tta.__all__)

    def test_triton_spec_in_all(self):
        self.assertIn("TritonSpec", tta.__all__)

    def test_cutile_spec_in_all(self):
        self.assertIn("CuTileSpec", tta.__all__)

    def test_cutedsl_spec_in_all(self):
        self.assertIn("CuTeDSLSpec", tta.__all__)

    def test_plugin_impl_descriptor_in_all(self):
        self.assertIn("CustomPluginSpec", tta.__all__)

    def test_self_attr_in_all(self):
        self.assertIn("self_attr", tta.__all__)


# ---------------------------------------------------------------------------
# TestGetAnnotationMetadata
# ---------------------------------------------------------------------------

class TestGetAnnotationMetadata(unittest.TestCase):
    """get_annotation_metadata helper returns the attached AnnotationMetadata."""

    def test_returns_none_for_unannotated_function(self):
        def plain(x):
            return x
        result = get_annotation_metadata(plain)
        self.assertIsNone(result)

    def test_returns_metadata_for_annotated_function(self):
        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op(x):
            return x * 2
        md = get_annotation_metadata(op)
        self.assertIsNotNone(md)
        self.assertIsInstance(md, AnnotationMetadata)

    def test_metadata_impl_matches_spec(self):
        spec = tta.plugin("MyPlugin", "2.0", "myns")
        @tta.export_as(impl=spec)
        def op(x):
            return x
        md = get_annotation_metadata(op)
        self.assertIsInstance(md.impl, tta.RegistryPluginSpec)
        self.assertEqual(md.impl.name, "MyPlugin")

    def test_metadata_name_preserved(self):
        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"), name="preserved_name")
        def op(x):
            return x
        md = get_annotation_metadata(op)
        self.assertEqual(md.name, "preserved_name")

    def test_metadata_require_preserved(self):
        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"), require=True)
        def op(x):
            return x
        md = get_annotation_metadata(op)
        self.assertTrue(md.require)

    def test_metadata_survives_functools_wraps(self):
        """get_annotation_metadata works on functions re-wrapped with @functools.wraps.

        functools.wraps copies fn.__dict__ to wrapper.__dict__; since
        attach_annotation_metadata stores metadata in fn.__dict__, the copy
        makes the metadata available on the outer wrapper too.
        """
        import functools

        def outer(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper

        @outer
        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op(x):
            return x

        # The outer wrapper should carry the metadata via __dict__ copy.
        md = get_annotation_metadata(op)
        self.assertIsNotNone(md, "metadata should be accessible on functools.wraps wrapper")


# ---------------------------------------------------------------------------
# TestSelfAttrSpec
# ---------------------------------------------------------------------------

class TestSelfAttrSpec(unittest.TestCase):
    """tta.self_attr() returns a FromMethodSelf with the correct attr_name."""

    def test_self_attr_returns_from_method_self(self):
        fms = tta.self_attr("conv.weight")
        self.assertIsInstance(fms, FromMethodSelf)

    def test_self_attr_stores_path(self):
        fms = tta.self_attr("layer.sub.bias")
        self.assertEqual(fms.attr_name, "layer.sub.bias")

    def test_self_attr_single_level_path(self):
        fms = tta.self_attr("alpha")
        self.assertEqual(fms.attr_name, "alpha")

    def test_self_attr_as_default_in_function_signature(self):
        """self_attr used as a default param value is detectable via __wrapped__."""
        import inspect

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op(self, x, w=tta.self_attr("weight")):
            return x * w

        # export_as decorator uses @functools.wraps so op.__wrapped__ is the original fn.
        original_fn = getattr(op, "__wrapped__", op)
        params = inspect.signature(original_fn).parameters
        w_param = params.get("w")
        self.assertIsNotNone(w_param, "expected 'w' parameter in original function signature")
        self.assertIsInstance(w_param.default, FromMethodSelf)
        self.assertEqual(w_param.default.attr_name, "weight")

    def test_self_attr_eager_resolution_on_instance_method(self):
        """An export_as-decorated instance method with self_attr runs correctly in eager mode.

        In eager mode the body executes directly; self_attr defaults are never
        invoked as real arguments (the body uses self directly).
        """

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 3.0

            @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
            def scale_op(self, x, s=tta.self_attr("scale")):
                return x * self.scale  # uses self directly in eager

            def forward(self, x):
                return self.scale_op(x)

        m = M()
        x = torch.randn(4)
        out = m(x)
        expected = x * 3.0
        torch.testing.assert_close(out, expected)

    def test_multiple_self_attrs_in_one_function(self):
        """Multiple self_attr defaults are all FromMethodSelf instances."""
        import inspect

        @tta.export_as(impl=tta.plugin("P", "1.0", "ns"))
        def op(self, x, w=tta.self_attr("weight"), b=tta.self_attr("bias")):
            return x * w + b

        # export_as uses @functools.wraps; original fn is accessible via __wrapped__.
        original_fn = getattr(op, "__wrapped__", op)
        params = inspect.signature(original_fn).parameters
        for param_name in ("w", "b"):
            p = params.get(param_name)
            self.assertIsNotNone(p, f"expected param '{param_name}' in signature")
            self.assertIsInstance(p.default, FromMethodSelf)


if __name__ == "__main__":
    unittest.main()
