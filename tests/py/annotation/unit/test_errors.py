"""Unit tests for TTA error types."""

import unittest

from torch_tensorrt.annotation._errors import TTADiagnosticError


class TestTTADiagnosticError(unittest.TestCase):
    def test_is_runtime_error(self):
        err = TTADiagnosticError("boom", stage="build")
        self.assertIsInstance(err, RuntimeError)

    def test_stage_stored(self):
        err = TTADiagnosticError("msg", stage="export")
        self.assertEqual(err.stage, "export")

    def test_message_stored(self):
        err = TTADiagnosticError("something broke", stage="lowering")
        self.assertEqual(err.message, "something broke")

    def test_leaf_op_none_by_default(self):
        err = TTADiagnosticError("msg", stage="build")
        self.assertIsNone(err.leaf_op)

    def test_impl_id_none_by_default(self):
        err = TTADiagnosticError("msg", stage="build")
        self.assertIsNone(err.impl_id)

    def test_leaf_op_stored(self):
        err = TTADiagnosticError("msg", stage="build", leaf_op="aten::linear")
        self.assertEqual(err.leaf_op, "aten::linear")

    def test_impl_id_stored(self):
        err = TTADiagnosticError("msg", stage="build", impl_id="my_kernel")
        self.assertEqual(err.impl_id, "my_kernel")

    def test_str_contains_stage_uppercased(self):
        err = TTADiagnosticError("msg", stage="export")
        self.assertIn("EXPORT", str(err))

    def test_str_contains_message(self):
        err = TTADiagnosticError("detailed error text", stage="build")
        self.assertIn("detailed error text", str(err))

    def test_str_contains_leaf_op(self):
        err = TTADiagnosticError("msg", stage="build", leaf_op="ns::op")
        self.assertIn("ns::op", str(err))

    def test_str_contains_impl_id(self):
        err = TTADiagnosticError("msg", stage="build", impl_id="impl_abc")
        self.assertIn("impl_abc", str(err))

    def test_str_no_leaf_op_no_impl_id(self):
        err = TTADiagnosticError("pure msg", stage="lowering")
        s = str(err)
        self.assertIn("[TTA LOWERING]", s)
        self.assertIn("pure msg", s)
        self.assertNotIn("impl=", s)

    def test_str_with_all_fields(self):
        err = TTADiagnosticError("full", stage="build", leaf_op="op", impl_id="impl")
        s = str(err)
        self.assertIn("[TTA BUILD]", s)
        self.assertIn("op", s)
        self.assertIn("impl=impl", s)
        self.assertIn("full", s)

    def test_can_be_raised_and_caught(self):
        with self.assertRaises(TTADiagnosticError) as ctx:
            raise TTADiagnosticError("raised", stage="export")
        self.assertEqual(ctx.exception.stage, "export")

    def test_can_be_caught_as_runtime_error(self):
        with self.assertRaises(RuntimeError):
            raise TTADiagnosticError("raised", stage="build")


if __name__ == "__main__":
    unittest.main()
