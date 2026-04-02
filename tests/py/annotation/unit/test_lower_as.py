"""Unit tests for tta.lower_as API and helpers.

Tests cover:
- TestLowerAsAPI        — eager no-op, capture-mode region recording, impl_id helpers.
- TestIOCompatibility   — io_compatible() return type, input/output count mismatch,
                          no-arity and io_signatures-based checks; LowerAsError structured fields.
- TestLowerAsEagerNoOp  — end-to-end eager execution is numerically identical with/without annotation.
"""

import unittest

import tensorrt as trt
import torch
import torch.nn as nn

import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation._capture_state import (
    active_regions,
    in_capture_mode,
    next_region_id,
    set_capture_mode,
)
from torch_tensorrt.annotation._lower_as_api import (
    clear_lower_as_regions,
    get_all_lower_as_regions,
    lower_as,
)
from torch_tensorrt.annotation.ir.region_discovery import BoundaryTensor, RegionIO
from torch_tensorrt.annotation._lower_as_api import spec_to_impl_id
from torch_tensorrt.annotation.ir.boundary_validation import io_compatible
from torch_tensorrt.annotation import LowerAsError


class TestLowerAsAPI(unittest.TestCase):
    def tearDown(self):
        set_capture_mode(False)
        clear_lower_as_regions()

    def test_lower_as_eager_no_effect(self):
        set_capture_mode(False)
        impl = tta.builtin("add_activation", type=1)
        with tta.lower_as(impl=impl, require=True, name="x"):
            pass
        self.assertEqual(len(get_all_lower_as_regions()), 0)

    def test_lower_as_capture_records_region(self):
        set_capture_mode(True)
        impl = tta.builtin("add_activation", type=1)
        with tta.lower_as(impl=impl, require=False, name="my_region"):
            self.assertEqual(len(active_regions()), 1)
        self.assertEqual(len(active_regions()), 0)
        regions = get_all_lower_as_regions()
        self.assertEqual(len(regions), 1)
        cfg = list(regions.values())[0]
        self.assertEqual(cfg.name, "my_region")
        self.assertFalse(cfg.require)
        self.assertIs(cfg.impl, impl)

    def test_spec_to_impl_id_builtin(self):
        impl = tta.builtin("add_activation", type=1)
        uid = spec_to_impl_id(impl)
        self.assertIsInstance(uid, str)
        self.assertIn("add_activation", uid)
        # Same spec constructed with identical args → same ID (deterministic cache key).
        impl2 = tta.builtin("add_activation", type=1)
        self.assertEqual(spec_to_impl_id(impl), spec_to_impl_id(impl2))
        # Different spec → different ID.
        other = tta.plugin("MyPlugin", "1", "")
        self.assertNotEqual(uid, spec_to_impl_id(other))

    def test_spec_to_impl_id_plugin(self):
        impl = tta.plugin("MyPlugin", "1", "")
        uid = spec_to_impl_id(impl)
        self.assertIsInstance(uid, str)
        self.assertIn("MyPlugin", uid)
        # Same plugin name/version/namespace → same ID.
        impl2 = tta.plugin("MyPlugin", "1", "")
        self.assertEqual(uid, spec_to_impl_id(impl2))
        # Different plugin name → different ID.
        other = tta.plugin("OtherPlugin", "1", "")
        self.assertNotEqual(uid, spec_to_impl_id(other))

    def test_spec_to_impl_id_unknown_type_uses_id_fallback(self):
        """Unknown spec types fall back to str(id(impl)) — a non-empty string."""
        obj = object()
        uid = spec_to_impl_id(obj)
        self.assertIsInstance(uid, str)
        self.assertTrue(len(uid) > 0)

    def test_spec_to_impl_id_two_distinct_objects_differ(self):
        """Two different unknown objects get distinct IDs (within the same process)."""
        obj_a = object()
        obj_b = object()
        self.assertNotEqual(spec_to_impl_id(obj_a), spec_to_impl_id(obj_b))

    def test_lower_as_used_as_decorator(self):
        """lower_as can wrap a function via decorator syntax (ContextDecorator)."""
        set_capture_mode(False)
        impl = tta.builtin("add_activation", type=1)

        @tta.lower_as(impl=impl, require=False)
        def body(x):
            return x + 1

        x = torch.tensor([2.0])
        result = body(x)
        torch.testing.assert_close(result, x + 1)

    def test_lower_as_decorator_in_capture_records_region(self):
        """lower_as decorator in capture mode records a lower_as region."""
        set_capture_mode(True)
        impl = tta.builtin("add_activation", type=1)
        clear_lower_as_regions()

        @tta.lower_as(impl=impl, require=False, name="deco_region")
        def body(x):
            return x + 1

        x = torch.tensor([1.0])
        body(x)
        regions = get_all_lower_as_regions()
        self.assertEqual(len(regions), 1)
        cfg = list(regions.values())[0]
        self.assertEqual(cfg.name, "deco_region")


# ---------------------------------------------------------------------------
# Helpers for TestIOCompatibility
# ---------------------------------------------------------------------------

def _make_tensor_meta(dtype=torch.float32, shape=(2, 3)):
    """Return a simple namespace mimicking tensor_meta."""
    class _TM:
        pass
    tm = _TM()
    tm.dtype = dtype
    tm.shape = shape
    return tm


def _make_bt(dtype=torch.float32, shape=(2, 3)):
    """Build a BoundaryTensor with no real FX node."""
    return BoundaryTensor(node=None, tensor_meta=_make_tensor_meta(dtype, shape))


def _make_region_io(n_in=2, n_out=1):
    """Build a RegionIO with dummy BoundaryTensors."""
    inputs = [_make_bt() for _ in range(n_in)]
    outputs = [_make_bt() for _ in range(n_out)]
    return RegionIO(inputs=inputs, outputs=outputs)


class _ArityImpl:
    """Mock impl descriptor with a fixed arity (num_inputs / num_outputs)."""

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


class _NoArityImpl:
    """Mock impl descriptor with no arity information at all."""
    pass


class _SigImpl:
    """Mock impl descriptor with an io_signatures() method."""

    class _Sig:
        def __init__(self, num_inputs, num_outputs):
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs

    def __init__(self, num_inputs, num_outputs):
        self._sig = self._Sig(num_inputs, num_outputs)

    def io_signatures(self):
        return [self._sig]


# ---------------------------------------------------------------------------

class TestIOCompatibility(unittest.TestCase):
    """Tests for io_compatible() and LowerAsError structured fields."""

    # ------------------------------------------------------------------
    # io_compatible return type
    # ------------------------------------------------------------------

    def test_io_compatible_returns_tuple(self):
        """io_compatible must return a 2-tuple (bool, str)."""
        region = _make_region_io(n_in=1, n_out=1)
        impl = _NoArityImpl()
        result = io_compatible(region, impl)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        ok, reason = result
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reason, str)

    # ------------------------------------------------------------------
    # Always-on count check
    # ------------------------------------------------------------------

    def test_count_mismatch_raises(self):
        """Region with 2 inputs but impl expects 1 → LowerAsError on require=True."""
        region = _make_region_io(n_in=2, n_out=1)
        impl = _ArityImpl(num_inputs=1, num_outputs=1)
        ok, reason = io_compatible(region, impl)
        self.assertFalse(ok)
        self.assertIn("input count", reason)
        self.assertIn("region=2", reason)
        self.assertIn("impl=1", reason)

    def test_output_count_mismatch(self):
        """Region with 2 outputs but impl expects 1 → not compatible."""
        region = _make_region_io(n_in=1, n_out=2)
        impl = _ArityImpl(num_inputs=1, num_outputs=1)
        ok, reason = io_compatible(region, impl)
        self.assertFalse(ok)
        self.assertIn("output count", reason)
        self.assertIn("region=2", reason)
        self.assertIn("impl=1", reason)

    def test_matching_counts_no_raise(self):
        """Matching input/output counts → compatible (no raise)."""
        region = _make_region_io(n_in=2, n_out=1)
        impl = _ArityImpl(num_inputs=2, num_outputs=1)
        ok, reason = io_compatible(region, impl)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_no_arity_info_is_compatible(self):
        """If impl provides no arity info, any count is accepted."""
        region = _make_region_io(n_in=5, n_out=3)
        impl = _NoArityImpl()
        ok, _ = io_compatible(region, impl)
        self.assertTrue(ok)

    # ------------------------------------------------------------------
    # io_signatures-based check
    # ------------------------------------------------------------------

    def test_sig_count_mismatch(self):
        """Mismatch via io_signatures → not compatible."""
        region = _make_region_io(n_in=2, n_out=1)
        impl = _SigImpl(num_inputs=1, num_outputs=1)
        ok, reason = io_compatible(region, impl)
        self.assertFalse(ok)
        self.assertIn("input count", reason)
        self.assertIn("region=2", reason)

    def test_sig_matching_counts_compatible(self):
        """Matching io_signatures counts → compatible."""
        region = _make_region_io(n_in=2, n_out=1)
        impl = _SigImpl(num_inputs=2, num_outputs=1)
        ok, reason = io_compatible(region, impl)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    # ------------------------------------------------------------------
    # LowerAsError structured fields
    # ------------------------------------------------------------------

    def test_lower_as_error_has_structured_fields(self):
        """LowerAsError should carry rid, name, reason attributes."""
        err = LowerAsError("test error", rid=42, name="encoder", reason="input count: region=2, impl=1")
        self.assertEqual(err.rid, 42)
        self.assertEqual(err.name, "encoder")
        self.assertEqual(err.reason, "input count: region=2, impl=1")
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("test error", str(err))

    def test_lower_as_error_defaults_none(self):
        """LowerAsError fields default to None when not provided."""
        err = LowerAsError("plain error")
        self.assertIsNone(err.rid)
        self.assertIsNone(err.name)
        self.assertIsNone(err.reason)


# ---------------------------------------------------------------------------
# TestLowerAsEagerNoOp (moved from integration/test_lower_as_e2e.py)
# ---------------------------------------------------------------------------

class TestLowerAsEagerNoOp(unittest.TestCase):
    def test_lower_as_eager_noop(self):
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU), require=False):
                    x = torch.relu(x)
                return x

        m = M()
        x = torch.randn(2, 8)
        with torch.no_grad():
            out_with = m(x.clone())
        with torch.no_grad():
            out_without = torch.relu(x.clone())
        torch.testing.assert_close(out_with, out_without)
