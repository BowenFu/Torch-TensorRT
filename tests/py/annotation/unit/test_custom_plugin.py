"""Custom plugin pipeline tests (sandbox, tactics, registration, lowering)."""

import unittest
from unittest.mock import Mock

import torch

import torch_tensorrt.annotation as tta
from torch_tensorrt.annotation._custom_plugin._descriptor import (
    CustomPluginSpec,
    custom_plugin,
    lower_custom_plugin_descriptor,
    register_custom_plugin,
)
from torch_tensorrt.annotation._custom_plugin._lowering import lower_custom_plugin
from torch_tensorrt.annotation._custom_plugin._qdp_utils import (
    TTAPluginError,
    TacticEntry,
    build_tactic_table,
    derive_impl_id,
    fingerprint_fn,
    is_cutedsl_compile_fn,
    is_cutile_program,
    is_triton_kernel,
    make_qdp_symbol,
    make_sandboxed_host,
)
from torch_tensorrt.annotation._recorders import (
    CuTeRunRecorder,
    CuTileLaunchRecorder,
    TritonLaunchRecorder,
)
from torch_tensorrt.annotation._specs import CuTeDSLSpec, CuTileSpec, TritonSpec
from torch_tensorrt.annotation._custom_plugin._symbolic import SymbolicTensor, TensorRole


class TestTTAPluginError(unittest.TestCase):
    """Tests for TTAPluginError exception."""

    def test_tta_plugin_error_basic(self):
        """Test basic TTAPluginError creation with positional args."""
        err = TTAPluginError(
            op="test_op",
            stage="export",
            backend="triton",
            msg="Test error",
        )

        self.assertEqual(err.op, "test_op")
        self.assertEqual(err.stage, "export")
        self.assertEqual(err.backend, "triton")
        self.assertEqual(err.msg, "Test error")

    def test_tta_plugin_error_message_format(self):
        """Test TTAPluginError message format: '{op}: [{stage}] [{backend}] {msg}'."""
        err = TTAPluginError(
            op="my_kernel",
            stage="compile",
            backend="cutile",
            msg="Something went wrong",
        )

        err_str = str(err)
        self.assertIn("my_kernel", err_str)
        self.assertIn("compile", err_str)
        self.assertIn("cutile", err_str)
        self.assertIn("Something went wrong", err_str)

    def test_tta_plugin_error_is_runtime_error(self):
        """TTAPluginError is a RuntimeError subclass."""
        err = TTAPluginError(
            op="op",
            stage="stage",
            backend="backend",
            msg="msg",
        )
        self.assertIsInstance(err, RuntimeError)

    def test_tta_plugin_error_can_be_raised_and_caught(self):
        """TTAPluginError can be raised and caught."""
        with self.assertRaises(TTAPluginError) as ctx:
            raise TTAPluginError(
                op="my_op",
                stage="aot_impl",
                backend="triton",
                msg="kernel launch failed",
            )
        self.assertEqual(ctx.exception.op, "my_op")


class TestTacticTable(unittest.TestCase):
    """Tests for TacticEntry and build_tactic_table."""

    def test_tactic_entry_fields(self):
        """TacticEntry stores spec_idx and config_idx."""
        entry = TacticEntry(spec_idx=1, config_idx=2)
        self.assertEqual(entry.spec_idx, 1)
        self.assertEqual(entry.config_idx, 2)

    def test_build_tactic_table_single_spec_no_configs(self):
        """Single spec with no configs -> 1 tactic with empty config."""

        def kernel(x, out):
            pass

        spec = TritonSpec(launch_fn=kernel)
        table = build_tactic_table([spec])

        self.assertEqual(len(table), 1)
        self.assertEqual(table[0].spec_idx, 0)
        self.assertEqual(table[0].config_idx, 0)

    def test_build_tactic_table_single_spec_multiple_configs(self):
        """Single spec with N configs -> N tactics."""

        def kernel(x, out, BLOCK: int):
            pass

        spec = TritonSpec(launch_fn=kernel, configs=[{"BLOCK": 64}, {"BLOCK": 128}])
        table = build_tactic_table([spec])

        self.assertEqual(len(table), 2)
        self.assertEqual(table[0].config_idx, 0)
        self.assertEqual(table[1].config_idx, 1)

    def test_build_tactic_table_multiple_specs(self):
        """Multiple specs -> combined tactic table."""

        def k1(x, out):
            pass

        def k2(x, out):
            pass

        specs = [
            TritonSpec(launch_fn=k1, configs=[{"B": 64}, {"B": 128}]),
            CuTileSpec(launch_fn=k2),
        ]
        table = build_tactic_table(specs)

        self.assertEqual(len(table), 3)
        self.assertEqual(table[0].spec_idx, 0)
        self.assertEqual(table[1].spec_idx, 0)
        self.assertEqual(table[2].spec_idx, 1)

    def test_build_tactic_table_empty(self):
        """Empty spec list -> empty table."""
        table = build_tactic_table([])
        self.assertEqual(len(table), 0)

    def test_build_tactic_table_all_three_backends(self):
        """All three backends in one tactic table."""

        def f(x, out):
            pass

        table = build_tactic_table([
            tta.triton(f),
            tta.cutile(f),
            tta.cutedsl(f, arch="sm_80"),
        ])

        self.assertEqual(len(table), 3)
        for i in range(3):
            self.assertEqual(table[i].spec_idx, i)
            self.assertEqual(table[i].config_idx, 0)

    def test_build_tactic_table_via_descriptor(self):
        """CustomPluginSpec's specs produce correct tactic table."""

        def kernel(x, out, B: int):
            pass

        descriptor = tta.custom_plugin(
            tta.triton(kernel, configs=[{"B": 32}, {"B": 64}])
        )
        table = build_tactic_table(descriptor.specs)

        self.assertEqual(len(table), 2)

    def test_build_tactic_table_mixed_backends_configs(self):
        """Mixed backends with varying config counts."""

        def k1(x, out, B: int):
            pass

        def k2(x, out):
            pass

        def k3(x, out, T: int):
            pass

        descriptor = tta.custom_plugin([
            tta.triton(k1, configs=[{"B": 64}, {"B": 128}]),
            tta.cutile(k2),
            tta.cutedsl(k3, arch="sm_80", configs=[{"T": 32}, {"T": 64}]),
        ])

        table = build_tactic_table(descriptor.specs)
        self.assertEqual(len(table), 5)


class TestMakeSandboxedHost(unittest.TestCase):
    """Tests for make_sandboxed_host."""

    def test_sandboxed_host_returns_callable(self):
        """make_sandboxed_host returns a callable."""

        def kernel(x, out):
            pass

        result = make_sandboxed_host(kernel, {})
        self.assertTrue(callable(result))

    def test_sandboxed_host_with_overrides(self):
        """make_sandboxed_host injects overrides into the function globals."""
        _sentinel = object()

        def kernel(x, out):
            return _sentinel_global  # noqa: F821

        sandboxed = make_sandboxed_host(kernel, {"_sentinel_global": _sentinel})

        result = sandboxed(None, None)
        self.assertIs(result, _sentinel)

    def test_sandboxed_host_preserves_name(self):
        """make_sandboxed_host preserves the function name."""

        def my_custom_kernel(x, out):
            pass

        sandboxed = make_sandboxed_host(my_custom_kernel, {})
        self.assertEqual(sandboxed.__name__, "my_custom_kernel")

    def test_sandboxed_host_does_not_shadow_originals(self):
        """make_sandboxed_host does not mutate the original launch_fn's globals."""

        def kernel(x, out):
            pass

        orig_globals_id = id(kernel.__globals__)
        make_sandboxed_host(kernel, {"_injected": True})

        self.assertNotIn("_injected", kernel.__globals__)


class TestBackendDetection(unittest.TestCase):
    """Tests for backend heuristic detection functions."""

    def test_is_triton_kernel_plain_callable(self):
        """Plain callables are not Triton kernels."""

        def fn(x):
            pass

        self.assertFalse(is_triton_kernel(fn))

    def test_is_triton_kernel_returns_bool(self):
        """is_triton_kernel always returns a bool."""

        class MockObj:
            __module__ = "triton.runtime"
            def __getitem__(self, g):
                return self

        self.assertIsInstance(is_triton_kernel(MockObj()), bool)

    def test_is_cutile_program_plain_callable(self):
        """Plain callables are not cuTILE programs."""

        def fn(x):
            pass

        self.assertFalse(is_cutile_program(fn))

    def test_is_cutile_program_returns_bool(self):
        """is_cutile_program always returns a bool."""

        class MockObj:
            __module__ = "cuda.tile.program"
            def __call__(self):
                pass

        self.assertIsInstance(is_cutile_program(MockObj()), bool)

    def test_is_cutedsl_compile_fn_plain_callable(self):
        """Plain callables are not CuTe DSL compile functions."""

        def fn():
            pass

        self.assertFalse(is_cutedsl_compile_fn(fn))

    def test_is_cutedsl_compile_fn_returns_bool(self):
        """is_cutedsl_compile_fn always returns a bool."""

        class MockObj:
            __name__ = "compile"
            __module__ = "cute.dsl"

        self.assertIsInstance(is_cutedsl_compile_fn(MockObj()), bool)

    def test_non_callable_returns_false(self):
        """Non-callable objects return False from all detection functions."""
        for fn in [is_triton_kernel, is_cutile_program]:
            with self.subTest(fn=fn.__name__):
                result = fn(42)
                self.assertIsInstance(result, bool)


class TestFingerprinting(unittest.TestCase):
    """Tests for fingerprint_fn, derive_impl_id, and make_qdp_symbol."""

    def test_fingerprint_fn_returns_bytes(self):
        """fingerprint_fn returns bytes."""

        def kernel(x, out):
            pass

        fp = fingerprint_fn(kernel)
        self.assertIsInstance(fp, bytes)
        self.assertGreater(len(fp), 0)

    def test_fingerprint_fn_is_deterministic(self):
        """Same function produces same fingerprint each call."""

        def kernel(x, out):
            pass

        self.assertEqual(fingerprint_fn(kernel), fingerprint_fn(kernel))

    def test_fingerprint_fn_different_for_different_fns(self):
        """Different functions produce different fingerprints."""

        def kernel_a(x, out):
            pass

        def kernel_b(x, out, BLOCK: int):
            pass

        self.assertNotEqual(fingerprint_fn(kernel_a), fingerprint_fn(kernel_b))

    def test_derive_impl_id_returns_hex_string(self):
        """derive_impl_id returns a hex string."""

        def kernel(x, out):
            pass

        spec = TritonSpec(launch_fn=kernel)
        impl_id = derive_impl_id([spec])

        self.assertIsInstance(impl_id, str)
        self.assertTrue(all(c in "0123456789abcdef" for c in impl_id))

    def test_derive_impl_id_is_deterministic(self):
        """Same specs produce same impl_id."""

        def kernel(x, out):
            pass

        spec = TritonSpec(launch_fn=kernel)
        self.assertEqual(derive_impl_id([spec]), derive_impl_id([spec]))

    def test_make_qdp_symbol_format(self):
        """make_qdp_symbol returns 'tta_custom::host_kernel_XXXXXXXX' format."""
        sym = make_qdp_symbol("abcdef1234567890")
        self.assertTrue(sym.startswith("tta_custom::host_kernel_"))
        self.assertIn("::", sym)

    def test_make_qdp_symbol_uses_first_8_chars(self):
        """make_qdp_symbol uses only the first 8 chars of impl_id."""
        sym = make_qdp_symbol("abcdef1234567890")
        suffix = sym.split("host_kernel_")[1]
        self.assertEqual(suffix, "abcdef12")


class TestNewLaunchRecorders(unittest.TestCase):
    """Tests for the new recorder classes in _recorders.py."""

    def test_triton_launch_recorder_construction(self):
        """TritonLaunchRecorder can be constructed with real_kernel=None."""
        rec = TritonLaunchRecorder(real_kernel=None)
        self.assertIsNone(rec.grid)
        self.assertIsNone(rec.args)

    def test_triton_launch_recorder_captures_grid_and_args(self):
        """TritonLaunchRecorder[grid](*args) records grid and args."""
        rec = TritonLaunchRecorder(real_kernel=None)
        launcher = rec[(4, 1, 1)]
        launcher(torch.tensor(1.0), torch.tensor(2.0))

        self.assertEqual(rec.grid, (4, 1, 1))
        self.assertIsNotNone(rec.args)
        self.assertEqual(len(rec.args), 2)

    def test_triton_launch_recorder_captures_kwargs(self):
        """TritonLaunchRecorder records kwargs passed to launcher."""
        rec = TritonLaunchRecorder(real_kernel=None)
        launcher = rec[(1,)]
        launcher(torch.tensor(1.0), num_warps=4)

        self.assertIsNotNone(rec.kwargs)
        self.assertEqual(rec.kwargs.get("num_warps"), 4)

    def test_cutile_launch_recorder_construction(self):
        """CuTileLaunchRecorder can be constructed with real_prog=None."""
        rec = CuTileLaunchRecorder(real_prog=None)
        self.assertIsNone(rec.args)

    def test_cutile_launch_recorder_captures_call(self):
        """CuTileLaunchRecorder captures call args."""
        rec = CuTileLaunchRecorder(real_prog=None)
        rec(torch.tensor(1.0), torch.tensor(2.0))

        self.assertIsNotNone(rec.args)
        self.assertEqual(len(rec.args), 2)

    def test_cute_run_recorder_construction(self):
        """CuTeRunRecorder can be constructed with defaults."""
        rec = CuTeRunRecorder()
        self.assertIsNone(rec.args)
        self.assertIsNone(rec.compiled)

    def test_cute_run_recorder_record_run(self):
        """CuTeRunRecorder.record_run captures compiled and args."""
        rec = CuTeRunRecorder()
        mock_compiled = object()
        rec.record_run(mock_compiled, torch.tensor(1.0))

        self.assertIs(rec.compiled, mock_compiled)
        self.assertIsNotNone(rec.args)
        self.assertEqual(len(rec.args), 1)


class TestSymbolicTensorAndTensorRole(unittest.TestCase):
    """Tests for TensorRole enum and SymbolicTensor construction."""

    def test_tensor_role_members(self):
        """TensorRole has INPUT and OUTPUT members."""
        self.assertIn("INPUT", TensorRole.__members__)
        self.assertIn("OUTPUT", TensorRole.__members__)

    def test_tensor_role_values_are_distinct(self):
        """TensorRole.INPUT != TensorRole.OUTPUT."""
        self.assertNotEqual(TensorRole.INPUT, TensorRole.OUTPUT)

    def test_symbolic_tensor_construction_with_mock_td(self):
        """SymbolicTensor can be constructed with a mock TensorDesc."""

        class MockTD:
            pass

        mock_td = MockTD()
        st = SymbolicTensor(td=mock_td, role=TensorRole.INPUT, index=0)

        self.assertIs(st.td, mock_td)
        self.assertEqual(st.role, TensorRole.INPUT)
        self.assertEqual(st.index, 0)

    def test_symbolic_tensor_output_role(self):
        """SymbolicTensor supports OUTPUT role."""

        class MockTD:
            pass

        st = SymbolicTensor(td=MockTD(), role=TensorRole.OUTPUT, index=1)
        self.assertEqual(st.role, TensorRole.OUTPUT)
        self.assertEqual(st.index, 1)


class TestAnalyzeLaunchArgsWithMocks(unittest.TestCase):
    """Tests for analyze_launch_args using mock SymbolicTensors."""

    def _make_sym(self, role, index):
        class MockTD:
            pass
        return SymbolicTensor(td=MockTD(), role=role, index=index)

    def test_analyze_launch_args_one_input_one_output(self):
        """1 input + 1 output -> param_binding_indices [0, 1]."""
        from torch_tensorrt.annotation._custom_plugin._qdp_utils import analyze_launch_args

        sym_in = self._make_sym(TensorRole.INPUT, 0)
        sym_out = self._make_sym(TensorRole.OUTPUT, 0)

        bindings, scalars = analyze_launch_args(
            args=[sym_in, sym_out],
            num_inputs=1,
            num_outputs=1,
            op="test_op",
            backend="triton",
        )

        self.assertEqual(bindings, [0, 1])
        self.assertEqual(scalars, [])

    def test_analyze_launch_args_two_inputs_one_output(self):
        """2 inputs + 1 output -> param_binding_indices [0, 1, 2]."""
        from torch_tensorrt.annotation._custom_plugin._qdp_utils import analyze_launch_args

        sym_in0 = self._make_sym(TensorRole.INPUT, 0)
        sym_in1 = self._make_sym(TensorRole.INPUT, 1)
        sym_out = self._make_sym(TensorRole.OUTPUT, 0)

        bindings, scalars = analyze_launch_args(
            args=[sym_in0, sym_in1, sym_out],
            num_inputs=2,
            num_outputs=1,
            op="test_op",
            backend="triton",
        )

        self.assertEqual(bindings, [0, 1, 2])
        self.assertEqual(scalars, [])

    def test_analyze_launch_args_with_scalar_int(self):
        """Integer scalar args are converted to SymInt32 or stored as-is."""
        from torch_tensorrt.annotation._custom_plugin._qdp_utils import analyze_launch_args

        sym_in = self._make_sym(TensorRole.INPUT, 0)
        sym_out = self._make_sym(TensorRole.OUTPUT, 0)

        bindings, scalars = analyze_launch_args(
            args=[sym_in, sym_out, 128],
            num_inputs=1,
            num_outputs=1,
            op="test_op",
            backend="triton",
        )

        self.assertEqual(len(bindings), 2)
        self.assertEqual(len(scalars), 1)

    def test_analyze_launch_args_tensor_after_scalar_raises(self):
        """Tensor arg after a scalar raises TTAPluginError."""
        from torch_tensorrt.annotation._custom_plugin._qdp_utils import analyze_launch_args

        sym_in = self._make_sym(TensorRole.INPUT, 0)
        sym_out = self._make_sym(TensorRole.OUTPUT, 0)

        with self.assertRaises(TTAPluginError):
            analyze_launch_args(
                args=[sym_in, 128, sym_out],
                num_inputs=1,
                num_outputs=1,
                op="test_op",
                backend="triton",
            )

    def test_analyze_launch_args_invalid_type_raises(self):
        """Unknown argument type raises TTAPluginError."""
        from torch_tensorrt.annotation._custom_plugin._qdp_utils import analyze_launch_args

        with self.assertRaises(TTAPluginError):
            analyze_launch_args(
                args=["not_a_sym_tensor"],
                num_inputs=0,
                num_outputs=0,
                op="test_op",
                backend="triton",
            )


# ---------------------------------------------------------------------------
# Registration (consolidated: covers all three backends)
# ---------------------------------------------------------------------------


class TestRegisterCustomPlugin(unittest.TestCase):
    """Tests for register_custom_plugin with all backend types."""

    def test_register_triton_does_not_raise(self):
        """register_custom_plugin completes without exception for Triton."""

        def kernel_to_register(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel_to_register))

        try:
            register_custom_plugin(descriptor, num_inputs=1)
        except Exception as e:
            self.fail(f"register_custom_plugin raised unexpectedly: {e}")

    def test_register_cutile_does_not_raise(self):
        """register_custom_plugin completes without exception for CuTile."""

        def prog_to_register(x, out):
            pass

        descriptor = tta.custom_plugin(tta.cutile(prog_to_register))

        try:
            register_custom_plugin(descriptor, num_inputs=1)
        except Exception as e:
            self.fail(f"register_custom_plugin raised unexpectedly: {e}")

    def test_register_cutedsl_does_not_raise(self):
        """register_custom_plugin completes without exception for CuTeDSL."""

        def flash_attn_kernel(q, k, v, out):
            pass

        descriptor = tta.custom_plugin(tta.cutedsl(flash_attn_kernel, arch="sm_90"))

        try:
            register_custom_plugin(descriptor, num_inputs=3)
        except Exception as e:
            self.fail(f"register_custom_plugin raised unexpectedly: {e}")

    def test_register_is_idempotent(self):
        """Calling register_custom_plugin twice with same op is a no-op."""

        def kernel_idem(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel_idem))

        register_custom_plugin(descriptor, num_inputs=1)
        register_custom_plugin(descriptor, num_inputs=1)

    def test_register_multiple_inputs(self):
        """register_custom_plugin works with multiple inputs."""

        def add_kernel(x, y, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(add_kernel))
        register_custom_plugin(descriptor, num_inputs=2)

    def test_registered_ops_deduplication(self):
        """Two descriptors with the same specs share the same QDP registration."""
        from torch_tensorrt.annotation._custom_plugin._descriptor import _qdp_registered_ops

        def shared_kernel(x, out):
            pass

        spec = tta.triton(shared_kernel)
        d1 = tta.custom_plugin(spec)
        d2 = tta.custom_plugin(spec)

        self.assertEqual(d1.op_name, d2.op_name)

        register_custom_plugin(d1, num_inputs=1)
        register_custom_plugin(d2, num_inputs=1)

        self.assertIn(d1.op_name, _qdp_registered_ops)


# ---------------------------------------------------------------------------
# Lowering pipeline (mock TRT context)
# ---------------------------------------------------------------------------


class TestCustomPluginLoweringPipeline(unittest.TestCase):
    """Tests for lower_custom_plugin pipeline (with mock TRT ctx)."""

    def test_lower_custom_plugin_wraps_exceptions_as_tta_plugin_error(self):
        """lower_custom_plugin wraps unexpected errors as TTAPluginError."""

        def kernel(x, out):
            pass

        spec = tta.custom_plugin(tta.triton(kernel))

        ctx = Mock()
        ctx.net = Mock()
        ctx.net.add_plugin_v3 = Mock(side_effect=RuntimeError("mock net"))

        with self.assertRaises(TTAPluginError):
            lower_custom_plugin(ctx, spec, [Mock()], "test_layer")

    def test_lower_custom_plugin_registers_plugin_before_lowering(self):
        """lower_custom_plugin registers the plugin op_name in QDP before attempting to lower."""

        def kernel_for_lower(x, out):
            pass

        descriptor = tta.custom_plugin(tta.triton(kernel_for_lower))

        from torch_tensorrt.annotation._custom_plugin._descriptor import _qdp_registered_ops

        ctx = Mock()
        ctx.net = Mock()
        ctx.net.add_plugin_v3 = Mock(side_effect=RuntimeError("test"))

        try:
            lower_custom_plugin_descriptor(ctx, descriptor, [Mock()], "test")
        except TTAPluginError:
            pass

        self.assertIn(descriptor.op_name, _qdp_registered_ops)


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------


class TestAotImplModules(unittest.TestCase):
    """Tests for AOT impl module structure and importability."""

    def test_aot_impl_triton_is_importable(self):
        """aot_impl_triton can be imported."""
        from torch_tensorrt.annotation._custom_plugin._aot._triton import aot_impl_triton

        self.assertTrue(callable(aot_impl_triton))

    def test_aot_impl_cutile_is_importable(self):
        """aot_impl_cutile can be imported."""
        from torch_tensorrt.annotation._custom_plugin._aot._cutile import aot_impl_cutile

        self.assertTrue(callable(aot_impl_cutile))

    def test_aot_impl_cutedsl_is_importable(self):
        """aot_impl_cutedsl can be imported."""
        from torch_tensorrt.annotation._custom_plugin._aot._cutedsl import aot_impl_cutedsl

        self.assertTrue(callable(aot_impl_cutedsl))

    def test_make_compile_wrapper_importable(self):
        """_make_compile_wrapper can be imported."""
        from torch_tensorrt.annotation._custom_plugin._aot._cutedsl import _make_compile_wrapper

        self.assertTrue(callable(_make_compile_wrapper))

    def test_symbolic_tensor_and_cdiv_importable(self):
        """SymbolicTensor and cdiv can be imported."""
        from torch_tensorrt.annotation._custom_plugin._symbolic import TensorRole, cdiv

        self.assertTrue(callable(cdiv))
        self.assertIn("INPUT", TensorRole.__members__)
        self.assertIn("OUTPUT", TensorRole.__members__)


# ---------------------------------------------------------------------------
# Error paths (spec validation errors not already in test_specs.py)
# ---------------------------------------------------------------------------


class TestCustomPluginErrorPaths(unittest.TestCase):
    """Error path tests for custom_plugin and export_as."""

    def test_export_as_rejects_unknown_impl_type(self):
        """export_as raises TypeError for unknown impl types."""
        with self.assertRaises(TypeError):
            tta.export_as(impl={"unknown": "type"})

    def test_cutile_spec_invalid_configs_raises(self):
        """CuTileSpec rejects non-list configs."""

        def prog(x, out):
            pass

        with self.assertRaises(TypeError):
            tta.CuTileSpec(launch_fn=prog, configs="bad")

    def test_cutedsl_spec_invalid_launch_fn_raises(self):
        """CuTeDSLSpec rejects non-callable launch_fn."""
        with self.assertRaises(TypeError):
            tta.CuTeDSLSpec(launch_fn=123, arch="sm_90")

    def test_invalid_kernel_in_list_raises_type_error(self):
        """custom_plugin rejects list containing non-spec types."""

        def kernel(x, out):
            pass

        triton_spec = tta.triton(kernel)

        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin([triton_spec, "invalid"])

        self.assertIn("CuTileSpec", str(ctx.exception))


class TestWeightBinding(unittest.TestCase):
    """Tests for tensor-kwarg weight binding in custom_plugin()."""

    def _make_kernel(self):
        def kernel(x, w, out):
            pass
        return kernel

    def test_tensor_kwarg_goes_to_weights_not_attrs(self):
        """Tensor-valued kwargs land in descriptor.weights, not descriptor.attrs."""
        kernel = self._make_kernel()
        w = torch.randn(4, 4)
        desc = tta.custom_plugin(tta.triton(kernel), w=w)
        self.assertIn("w", desc.weights)
        self.assertNotIn("w", desc.attrs)

    def test_scalar_kwarg_goes_to_attrs_not_weights(self):
        """Scalar-valued kwargs land in descriptor.attrs, not descriptor.weights."""
        def kernel(x, out):
            pass
        desc = tta.custom_plugin(tta.triton(kernel), scale=2.0)
        self.assertIn("scale", desc.attrs)
        self.assertNotIn("scale", desc.weights)

    def test_mixed_kwargs_split_correctly(self):
        """Tensor and scalar kwargs are split into weights and attrs respectively."""
        kernel = self._make_kernel()
        w = torch.randn(8)
        desc = tta.custom_plugin(tta.triton(kernel), w=w, scale=3.0)
        self.assertIn("w", desc.weights)
        self.assertIn("scale", desc.attrs)
        self.assertNotIn("w", desc.attrs)
        self.assertNotIn("scale", desc.weights)

    def test_weight_stored_in_descriptor(self):
        """The actual tensor is accessible via descriptor.weights."""
        kernel = self._make_kernel()
        w = torch.randn(4, 4)
        desc = tta.custom_plugin(tta.triton(kernel), w=w)
        self.assertIs(desc.weights["w"], w)

    def test_num_weights_changes_op_name(self):
        """Different num_weights produces a different op_name / fingerprint."""
        def kernel(x, out):
            pass
        w = torch.randn(4)
        no_weight = tta.custom_plugin(tta.triton(kernel))
        with_weight = tta.custom_plugin(tta.triton(kernel), w=w)
        self.assertNotEqual(no_weight.op_name, with_weight.op_name)

    def test_two_weights_different_from_one_weight(self):
        """Declaring two weights gives a different op_name than one weight."""
        def kernel(x, w1, w2, out):
            pass
        w1 = torch.randn(4)
        w2 = torch.randn(4)
        one = tta.custom_plugin(tta.triton(kernel), w1=w1)
        two = tta.custom_plugin(tta.triton(kernel), w1=w1, w2=w2)
        self.assertNotEqual(one.op_name, two.op_name)

    def test_weight_binding_descriptor_has_correct_num_weights(self):
        """CustomPluginSpec.weights dict has expected length."""
        def kernel(x, w1, w2, out):
            pass
        w1, w2 = torch.randn(4), torch.randn(4)
        desc = tta.custom_plugin(tta.triton(kernel), w1=w1, w2=w2)
        self.assertEqual(len(desc.weights), 2)

    def test_derive_impl_id_includes_num_weights(self):
        """derive_impl_id includes num_weights in the fingerprint."""
        def kernel(x, out):
            pass
        spec = TritonSpec(launch_fn=kernel)
        id_no_weights = derive_impl_id([spec], num_weights=0)
        id_one_weight = derive_impl_id([spec], num_weights=1)
        id_two_weights = derive_impl_id([spec], num_weights=2)
        self.assertNotEqual(id_no_weights, id_one_weight)
        self.assertNotEqual(id_one_weight, id_two_weights)


if __name__ == "__main__":
    unittest.main()
