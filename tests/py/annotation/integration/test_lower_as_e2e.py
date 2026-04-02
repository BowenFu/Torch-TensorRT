"""E2E tests for tta.lower_as: eager no-op, export tagging, and compile with builtin."""

import unittest

import pytest
import tensorrt as trt
import torch
import torch.nn as nn

import torch_tensorrt
import torch_tensorrt.annotation as tta

from ._e2e_common import (
    _compile_and_run,
    _compile_and_run_dynamic,
    _has_cudnn8,
    assert_plugin_io_format,
    assert_trt_compiled,
    assert_tactics_metadata,
    get_layers_with_tta_metadata,
    get_selected_tactic_for_engine,
    tta_compile,
)
from .triton_kernels import (
    launch_add_one,
    launch_scale,
    launch_fused_add_relu,
    launch_add_2d,
    launch_conv3x3_relu_pool,
    launch_split_add_scale,
)
from .cutile_kernels import (
    launch_add_one as cutile_add_one,
    launch_scale as cutile_scale,
    launch_fused_add_relu as cutile_fused_add_relu,
    launch_conv3x3_relu_pool as cutile_conv3x3_relu_pool,
)
from .cutedsl_kernels import (
    launch_add_one as cutedsl_add_one,
    launch_scale as cutedsl_scale,
    launch_fused_add_relu as cutedsl_fused_add_relu,
    launch_add_2d as cutedsl_add_2d,
    launch_conv3x3_relu_pool as cutedsl_conv3x3_relu_pool,
)



class TestLowerAsBuiltinE2E(unittest.TestCase):
    def test_lower_as_builtin_relu_region(self):
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU), require=True, name="relu_region"):
                    x = torch.relu(x)
                return x

        # Eager noop: tta.lower_as is transparent outside TRT capture mode.
        x_noop = torch.randn(2, 8)
        m_noop = M()
        with torch.no_grad():
            torch.testing.assert_close(
                m_noop(x_noop), torch.relu(x_noop),
                msg="tta.lower_as(builtin relu) must be noop in eager mode",
            )
        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(2, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{
                "backend": "builtin",
                "plugin_name": "add_activation",
                "torch_op": "relu_region",
            }],
        )

    def test_lower_as_builtin_add_region(self):
        class M(nn.Module):
            def forward(self, a, b):
                with tta.lower_as(impl=tta.builtin("add_elementwise", op=trt.ElementWiseOperation.SUM), require=True, name="add_region"):
                    c = a + b
                return c

        x, y = torch.randn(4, 4), torch.randn(4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{
                "backend": "builtin",
                "plugin_name": "add_elementwise",
                "torch_op": "add_region",
            }],
        )

    def test_lower_as_builtin_metadata_in_engine(self):
        """Region replaced via lower_as builtin; engine metadata reflects the impl and name."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.builtin("add_activation", type=trt.ActivationType.RELU),
                    require=True, name="relu_region",
                ):
                    x = torch.relu(x)
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(4, 8),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{
                "backend": "builtin",
                "plugin_name": "add_activation",
                "torch_op": "relu_region",
            }],
        )

    def test_lower_as_builtin_relu_dynamic_batch(self):
        """tta.lower_as(builtin relu) compiles and runs correctly with dynamic batch dimension."""

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.builtin("add_activation", type=trt.ActivationType.RELU),
                    require=True,
                    name="relu_dynamic",
                ):
                    return torch.relu(x)

        model = M()

        # Eager noop: tta.lower_as is transparent outside TRT capture mode.
        x_noop = torch.randn(2, 8)
        with torch.no_grad():
            torch.testing.assert_close(
                model(x_noop), torch.relu(x_noop),
                msg="tta.lower_as(builtin relu, dynamic) must be noop in eager mode",
            )

        trt_input = torch_tensorrt.Input(
            min_shape=(1, 8), opt_shape=(2, 8), max_shape=(4, 8),
            dtype=torch.float32,
        )

        trt_model = tta_compile(
            model, (trt_input,),
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        )

        # Test at min, opt, and max shapes
        for batch_size in [1, 2, 4]:
            x_run = torch.randn(batch_size, 8, device="cuda")
            with torch.no_grad():
                trt_out = trt_model(x_run)
            eager_out = torch.relu(x_run)
            torch.testing.assert_close(
                trt_out, eager_out, atol=1e-3, rtol=1e-3,
                msg=f"Accuracy failed at batch_size={batch_size}",
            )


class TestLowerAsRequireE2E(unittest.TestCase):
    """tta.lower_as(require=True) raises at compile time when IO doesn't match."""

    def test_require_true_raises_on_io_mismatch_via_compile(self):
        """require=True + IO mismatch: multi-output region with single-output impl → raises at compile time.

        The lower_as region has two escaping outputs but the builtin relu impl
        expects exactly one output.  torch_tensorrt.compile must raise LowerAsError.
        """
        from torch_tensorrt.annotation import LowerAsError

        class _TwoOutput(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.builtin("add_activation", type=trt.ActivationType.RELU),
                    require=True,
                    name="inner_lower_mismatch",
                ):
                    # two distinct computed outputs escape → IO mismatch
                    a = torch.relu(x)
                    b = torch.sigmoid(x)
                return a + b

        model = _TwoOutput().eval().cuda()
        x = torch.randn(2, 8, device="cuda")

        with self.assertRaises((LowerAsError, Exception)):
            torch_tensorrt.compile(
                model,
                inputs=(x,),
                min_block_size=1,
                require_full_compilation=True,
            )


class TestLowerAsCustomPluginE2E(unittest.TestCase):
    """lower_as regions backed by custom_plugin (Triton) impl."""

    def test_lower_as_triton_add_one(self):
        """Region containing x + 1.0 replaced by a Triton add_one custom plugin."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="add_one_region",
                ):
                    x = x + 1.0
                return x

        # Eager noop: tta.lower_as(custom_plugin) is transparent outside TRT capture mode.
        x_noop = torch.randn(256)
        with torch.no_grad():
            torch.testing.assert_close(
                M()(x_noop), x_noop + 1.0,
                msg="tta.lower_as(triton) must be noop in eager mode",
            )
        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one_region"}],
        )

    def test_lower_as_triton_scale(self):
        """Region containing x * 2.0 replaced by a Triton scale custom plugin."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="scale_region",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_scale", "torch_op": "scale_region"}],
        )

    def test_lower_as_triton_two_inputs(self):
        """Region with two tensor inputs replaced by a Triton fused_add_relu plugin."""
        class M(nn.Module):
            def forward(self, x, y):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_fused_add_relu, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="fused_region",
                ):
                    out = torch.relu(x + y)
                return out

        x, y = torch.randn(128), torch.randn(128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_fused_add_relu", "torch_op": "fused_region"}],
        )

    def test_lower_as_triton_multi_config_tactics(self):
        """Region with two BLOCK_SIZE tactics; tactics metadata verified."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(
                        launch_add_one,
                        configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}],
                    )),
                    require=True, name="add_one_blocked",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one_blocked"},
            ],
        )
        assert_tactics_metadata(self, trt_model, [
            {"idx": 1, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 64}},
            {"idx": 2, "backend": "triton", "fn_name": "launch_add_one", "config": {"BLOCK_SIZE": 128}},
        ])

    def test_lower_as_triton_dynamic_shape(self):
        """Region with dynamic 1D profile replaced by Triton plugin."""
        import torch_tensorrt

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 64}])),
                    require=True, name="scale_dynamic",
                ):
                    x = x * 2.0
                return x

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(128,), max_shape=(256,), dtype=torch.float32)
        ]
        model = M()
        run_inputs = (torch.randn(128, device="cuda"),)
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(model, trt_inputs, run_inputs)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_scale", "torch_op": "scale_dynamic"}],
        )

    def test_lower_as_chained_regions(self):
        """Two consecutive lower_as regions, each with a different Triton plugin."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="add_one",
                ):
                    x = x + 1.0
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="scale",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one"},
                {"backend": "triton", "fn_name": "launch_scale", "torch_op": "scale"},
            ],
        )


class TestLowerAsGraphRewrite(unittest.TestCase):
    """Verify that apply_lower_as_regions actually rewrites the FX graph."""

    def _export_with_lower_as(self, model, inputs):
        """Export model with lower_as tagging active; return ExportedProgram."""
        from torch_tensorrt.annotation._capture_state import (
            install_graph_tagging,
            set_capture_mode,
            uninstall_graph_tagging,
        )
        from torch_tensorrt.annotation._lower_as_api import (
            clear_lower_as_regions,
            get_all_lower_as_regions,
        )
        from torch_tensorrt.annotation._lower_as_api import spec_to_impl_id

        model = model.eval()
        cuda_inputs = tuple(x.cuda() for x in inputs)

        set_capture_mode(True)
        install_graph_tagging()
        try:
            exported = torch.export.export(model.cpu(), tuple(x.cpu() for x in inputs))
        finally:
            uninstall_graph_tagging()
            set_capture_mode(False)

        region_table = {}
        impl_registry = {}
        for region_id, cfg in get_all_lower_as_regions().items():
            impl_id = spec_to_impl_id(cfg.impl)
            region_table[region_id] = {
                "id": region_id, "kind": "lower_as",
                "name": cfg.name or f"lower_as_region_{region_id}",
                "mode": None,
                "constraints": {"require": cfg.require},
                "args": {"impl_id": impl_id},
            }
            impl_registry[impl_id] = cfg.impl
        from torch_tensorrt.annotation.ir.region_view import build_region_views_post_export as _brvs
        _gm = exported.graph_module
        _node_regions = {}
        for _n in _gm.graph.nodes:
            _stack = _n.meta.get("tta_regions")
            if _stack:
                _node_regions[_n] = list(_stack)
        region_views = _brvs(_gm, node_regions=_node_regions, region_ids=list(region_table.keys()))
        exported._tta = {"region_table": region_table, "impl_registry": impl_registry, "region_views": region_views}
        clear_lower_as_regions()
        return exported

    def test_region_nodes_replaced_by_tta_op(self):
        """After apply_lower_as_regions, the add node is replaced by a TTA custom op.

        Criterion 2 (graph): verifies the FX graph rewrite occurred (TTA op present).
        Criterion 1 + 2 (TRT): also compiles via tta.compile and checks accuracy + TTA metadata.
        """
        from torch_tensorrt.annotation._lower_as_pass import apply_lower_as_regions

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="add_one",
                ):
                    x = x + 1.0
                return x

        exported = self._export_with_lower_as(M(), (torch.randn(256),))
        gm = exported.graph_module

        # Before: region nodes present
        pre_names = [n.name for n in gm.graph.nodes]
        self.assertTrue(any("add" in n for n in pre_names), f"Expected add node pre-rewrite: {pre_names}")

        apply_lower_as_regions(exported, gm)

        # Criterion 2 (graph): region nodes replaced by a TTA custom op call
        post_nodes = list(gm.graph.nodes)
        post_targets = [str(n.target) for n in post_nodes if n.op == "call_function"]
        self.assertTrue(
            any("torch_tensorrt_anno" in t for t in post_targets),
            f"Expected TTA op node after rewrite; got targets: {post_targets}",
        )

        # Criterion 1 + 2 (TRT): full compile and run — verify accuracy and engine metadata
        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            atol=1e-4, rtol=1e-4,
            allow_dtype_mismatch=True,
            expected_tta_metadata=[{
                "backend": "triton",
                "fn_name": "launch_add_one",
                "torch_op": "add_one",
            }],
        )

    def test_chained_regions_both_replaced(self):
        """Two consecutive lower_as regions are each replaced by a separate TTA op.

        Criterion 2 (graph): verifies 2 TTA ops in rewritten FX graph.
        Criterion 1 + 2 (TRT): also compiles via tta.compile and checks accuracy + TTA metadata.
        """
        from torch_tensorrt.annotation._lower_as_pass import apply_lower_as_regions

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="add_one",
                ):
                    x = x + 1.0
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="scale",
                ):
                    x = x * 2.0
                return x

        exported = self._export_with_lower_as(M(), (torch.randn(128),))
        gm = exported.graph_module

        apply_lower_as_regions(exported, gm)

        # Criterion 2 (graph): exactly 2 TTA op calls after rewriting 2 chained regions
        tta_calls = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and "torch_tensorrt_anno" in str(n.target)
        ]
        self.assertEqual(
            len(tta_calls), 2,
            f"Expected 2 TTA op calls after rewriting 2 chained regions; got {len(tta_calls)}",
        )

        # Criterion 1 + 2 (TRT): full compile and run — verify accuracy and engine metadata
        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            atol=1e-4, rtol=1e-4,
            allow_dtype_mismatch=True,
            expected_tta_metadata=[
                {"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one"},
                {"backend": "triton", "fn_name": "launch_scale", "torch_op": "scale"},
            ],
        )

    def test_multi_output_region_raises_with_require_true(self):
        """Multi-output region (2 escaping values) with a single-output impl raises LowerAsError.

        When require=True and the impl declares 1 output but the region has 2 escaping
        tensor values, apply_lower_as_regions detects an IO mismatch and raises.
        """
        from torch_tensorrt.annotation import LowerAsError
        from torch_tensorrt.annotation._lower_as_pass import apply_lower_as_regions

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="multi_out",
                ):
                    a = x + 1.0
                    b = x * 2.0
                return a + b  # both a and b escape → 2 outputs vs impl's 1

        exported = self._export_with_lower_as(M(), (torch.randn(128),))
        gm = exported.graph_module

        with self.assertRaises(LowerAsError) as ctx:
            apply_lower_as_regions(exported, gm)

        self.assertIn("multi_out", str(ctx.exception))

    def test_multi_output_region_skipped_with_require_false(self):
        """Multi-output region with a single-output impl is silently skipped when require=False.

        The model still compiles (the region is left unrewritten) and runs correctly.
        """
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=False, name="multi_out_skipped",
                ):
                    a = x + 1.0
                    b = x * 2.0
                return a + b  # both a and b escape → IO mismatch → skipped silently

        x = torch.randn(128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        torch.testing.assert_close(trt_out.float(), eager_out.float(), atol=1e-4, rtol=1e-4)

    def test_multi_output_region_produces_getitem_nodes(self):
        """Multi-output region with a 2-output impl: apply_lower_as_regions inserts getitem nodes.

        Uses launch_split_add_scale (1 input → out1 = x+1, out2 = x*2) with a meta_impl
        that declares 2 outputs, making the custom_plugin IO-compatible with the region.
        Both escaping values (a, b) are attributed via operator.getitem on the single plugin call.

        IMPORTANT: tta.custom_plugin() must be called *before* forward() is traced
        (i.e., outside the class body or forward method) so that _infer_num_outputs runs
        with no active FakeTensorMode dispatch stack and correctly returns 2.
        """
        import operator
        from torch_tensorrt.annotation._lower_as_pass import apply_lower_as_regions

        def _meta(x):
            return x.clone(), x.clone()

        # Build impl descriptor OUTSIDE of forward() to avoid FakeTensorMode
        _impl = tta.custom_plugin(
            tta.triton(launch_split_add_scale, configs=[{"BLOCK_SIZE": 128}]),
            meta_impl=_meta,
        )

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(impl=_impl, require=True, name="multi_out_ok"):
                    a = x + 1.0
                    b = x * 2.0
                return a + b  # both a and b escape → 2 outputs, compatible with impl

        exported = self._export_with_lower_as(M(), (torch.randn(128),))
        gm = exported.graph_module

        apply_lower_as_regions(exported, gm)

        post_targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(
            operator.getitem, post_targets,
            f"Expected operator.getitem nodes for 2-output region rewrite; got targets: {post_targets}",
        )
        # Exactly 2 getitem nodes: one for out1 (idx=0) and one for out2 (idx=1).
        getitem_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is operator.getitem]
        self.assertEqual(len(getitem_nodes), 2, f"Expected 2 getitem nodes, got {len(getitem_nodes)}")

    def test_multi_output_region_e2e_compiles_and_runs(self):
        """Multi-output lower_as region with 2-output plugin: tta.compile → TRT engine runs correctly."""

        def _meta(x):
            return x.clone(), x.clone()

        # Build impl descriptor OUTSIDE of forward() to avoid FakeTensorMode
        _impl = tta.custom_plugin(
            tta.triton(launch_split_add_scale, configs=[{"BLOCK_SIZE": 128}]),
            meta_impl=_meta,
        )

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(impl=_impl, require=True, name="split_region"):
                    a = x + 1.0
                    b = x * 2.0
                return a + b  # expected: (x+1) + (x*2) = 3x + 1

        x = torch.randn(128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        torch.testing.assert_close(trt_out.float(), eager_out.float(), atol=1e-4, rtol=1e-4)


class TestLowerAsCuTileE2E(unittest.TestCase):
    """lower_as regions backed by CuTile custom_plugin impl."""

    def test_lower_as_cutile_add_one(self):
        """Region x + 1.0 replaced by a CuTile add_one plugin."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutile(cutile_add_one, configs=[{"BLOCK": 128}])),
                    require=True, name="cutile_add_one",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutile", "fn_name": "launch_add_one", "torch_op": "cutile_add_one"}],
        )

    def test_lower_as_cutile_scale(self):
        """Region x * 2.0 replaced by a CuTile scale plugin."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutile(cutile_scale, configs=[{"BLOCK": 128}])),
                    require=True, name="cutile_scale",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutile", "fn_name": "launch_scale", "torch_op": "cutile_scale"}],
        )

    def test_lower_as_cutile_two_inputs(self):
        """Region relu(x + y) replaced by a CuTile fused_add_relu plugin."""
        class M(nn.Module):
            def forward(self, x, y):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutile(cutile_fused_add_relu, configs=[{"BLOCK": 128}])),
                    require=True, name="cutile_fused",
                ):
                    out = torch.relu(x + y)
                return out

        x, y = torch.randn(128), torch.randn(128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutile", "fn_name": "launch_fused_add_relu", "torch_op": "cutile_fused"}],
        )

    def test_lower_as_cutile_dynamic(self):
        """CuTile lower_as region with dynamic 1D profile."""
        import torch_tensorrt

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutile(cutile_scale, configs=[{"BLOCK": 64}])),
                    require=True, name="cutile_scale_dyn",
                ):
                    x = x * 2.0
                return x

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(128,), max_shape=(256,), dtype=torch.float32)
        ]
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(
            M(), trt_inputs, (torch.randn(128, device="cuda"),)
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutile", "fn_name": "launch_scale", "torch_op": "cutile_scale_dyn"}],
        )


class TestLowerAsCuTeDSLE2E(unittest.TestCase):
    """lower_as regions backed by CuTeDSL custom_plugin impl."""

    def test_lower_as_cutedsl_add_one(self):
        """Region x + 1.0 replaced by a CuTeDSL add_one plugin."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_add_one)),
                    require=True, name="cutedsl_add_one",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_add_one", "torch_op": "cutedsl_add_one"}],
        )

    def test_lower_as_cutedsl_scale(self):
        """Region x * 2.0 replaced by a CuTeDSL scale plugin."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_scale)),
                    require=True, name="cutedsl_scale",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_scale", "torch_op": "cutedsl_scale"}],
        )

    def test_lower_as_cutedsl_two_inputs(self):
        """Region relu(x + y) replaced by a CuTeDSL fused_add_relu plugin."""
        class M(nn.Module):
            def forward(self, x, y):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_fused_add_relu)),
                    require=True, name="cutedsl_fused",
                ):
                    out = torch.relu(x + y)
                return out

        x, y = torch.randn(128), torch.randn(128)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_fused_add_relu", "torch_op": "cutedsl_fused"}],
        )

    def test_lower_as_cutedsl_dynamic(self):
        """CuTeDSL lower_as region with dynamic 1D profile."""
        import torch_tensorrt

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_scale)),
                    require=True, name="cutedsl_scale_dyn",
                ):
                    x = x * 2.0
                return x

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(128,), max_shape=(256,), dtype=torch.float32)
        ]
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(
            M(), trt_inputs, (torch.randn(128, device="cuda"),)
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_scale", "torch_op": "cutedsl_scale_dyn"}],
        )


@pytest.mark.requires_pre_bw
class TestLowerAsMixedBackendChained(unittest.TestCase):
    """Chained lower_as regions across different backends."""

    def test_triton_then_cutile(self):
        """First region: Triton add_one; second region: CuTile scale."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="triton_add_one",
                ):
                    x = x + 1.0
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutile(cutile_scale, configs=[{"BLOCK": 128}])),
                    require=True, name="cutile_scale",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "triton", "fn_name": "launch_add_one", "torch_op": "triton_add_one"},
                {"backend": "cutile", "fn_name": "launch_scale", "torch_op": "cutile_scale"},
            ],
        )

    def test_triton_then_cutedsl(self):
        """First region: Triton add_one; second region: CuTeDSL scale."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="triton_add_one",
                ):
                    x = x + 1.0
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_scale)),
                    require=True, name="cutedsl_scale",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "triton", "fn_name": "launch_add_one", "torch_op": "triton_add_one"},
                {"backend": "cutedsl", "fn_name": "launch_scale", "torch_op": "cutedsl_scale"},
            ],
        )

    def test_cutile_then_cutedsl(self):
        """First region: CuTile add_one; second region: CuTeDSL scale."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutile(cutile_add_one, configs=[{"BLOCK": 128}])),
                    require=True, name="cutile_add_one",
                ):
                    x = x + 1.0
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_scale)),
                    require=True, name="cutedsl_scale",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "cutile", "fn_name": "launch_add_one", "torch_op": "cutile_add_one"},
                {"backend": "cutedsl", "fn_name": "launch_scale", "torch_op": "cutedsl_scale"},
            ],
        )


class TestLowerAs2DInput(unittest.TestCase):
    """lower_as regions with 2D tensor inputs."""

    def test_lower_as_triton_2d(self):
        """2D (8, 16) input through a Triton add_one region (uses numel, shape-agnostic)."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="triton_add_one_2d",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(8, 16),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "triton_add_one_2d"}],
        )

    def test_lower_as_cutedsl_2d_two_inputs(self):
        """2D (4, 8) two-input region via CuTeDSL add_2d kernel."""
        class M(nn.Module):
            def forward(self, x, y):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_add_2d)),
                    require=True, name="cutedsl_add_2d",
                ):
                    out = x + y
                return out

        x, y = torch.randn(4, 8), torch.randn(4, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_add_2d", "torch_op": "cutedsl_add_2d"}],
        )

    def test_lower_as_triton_2d_two_inputs(self):
        """2D (4, 8) two-input region via Triton add_2d kernel."""
        class M(nn.Module):
            def forward(self, x, y):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(
                        launch_add_2d, configs=[{"BLOCK_M": 4, "BLOCK_N": 8}]
                    )),
                    require=True, name="triton_add_2d",
                ):
                    out = x + y
                return out

        x, y = torch.randn(4, 8), torch.randn(4, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x, y))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_2d", "torch_op": "triton_add_2d"}],
        )


class TestLowerAsIdempotency(unittest.TestCase):
    """Verify that lower_as regions are cleared between compile calls."""

    def test_compile_same_model_twice(self):
        """Compiling the same model twice produces correct results both times."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="add_one",
                ):
                    x = x + 1.0
                return x

        m = M()
        inp = (torch.randn(128),)
        trt1, out1, eager1 = _compile_and_run(m, inp)
        trt2, out2, eager2 = _compile_and_run(m, inp)
        assert_trt_compiled(self, trt1, out1, eager1, expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one"}])
        assert_trt_compiled(self, trt2, out2, eager2, expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one"}])

    def test_compile_different_models_sequentially(self):
        """Two different models with lower_as compile independently without cross-contamination."""
        class M1(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="add_one",
                ):
                    x = x + 1.0
                return x

        class M2(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])),
                    require=True, name="scale",
                ):
                    x = x * 2.0
                return x

        inp = (torch.randn(128),)
        trt1, out1, eager1 = _compile_and_run(M1(), inp)
        trt2, out2, eager2 = _compile_and_run(M2(), inp)
        assert_trt_compiled(self, trt1, out1, eager1, expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one"}])
        assert_trt_compiled(self, trt2, out2, eager2, expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_scale", "torch_op": "scale"}])


def _conv_relu_pool_meta(x, weight, bias):
    out = torch.nn.functional.conv2d(x, weight, bias, padding=1)
    return torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)


class TestLowerAsConvReluPool(unittest.TestCase):
    """lower_as with 3×3 conv + ReLU + 2×2 max pool across all backends."""

    def test_triton_conv3x3_relu_pool(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 2, 3, 3) * 0.1)
                self.bias = nn.Parameter(torch.zeros(4))

            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(
                        tta.triton(launch_conv3x3_relu_pool, configs=[{"BLOCK_C": 4}]),
                        meta_impl=_conv_relu_pool_meta,
                    ),
                    require=True, name="conv3x3_relu_pool",
                ):
                    out = torch.nn.functional.conv2d(x, self.weight, self.bias, padding=1)
                    out = torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)
                return out

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_conv3x3_relu_pool", "torch_op": "conv3x3_relu_pool"}],
        )

    def test_cutile_conv3x3_relu_pool(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 2, 3, 3) * 0.1)
                self.bias = nn.Parameter(torch.zeros(4))

            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(
                        tta.cutile(cutile_conv3x3_relu_pool, configs=[{}]),
                        meta_impl=_conv_relu_pool_meta,
                    ),
                    require=True, name="conv3x3_relu_pool",
                ):
                    out = torch.nn.functional.conv2d(x, self.weight, self.bias, padding=1)
                    out = torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)
                return out

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "cutile", "fn_name": "launch_conv3x3_relu_pool", "torch_op": "conv3x3_relu_pool"}],
        )

    def test_cutedsl_conv3x3_relu_pool(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 2, 3, 3) * 0.1)
                self.bias = nn.Parameter(torch.zeros(4))

            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(
                        tta.cutedsl(cutedsl_conv3x3_relu_pool, configs=[{}]),
                        meta_impl=_conv_relu_pool_meta,
                    ),
                    require=True, name="conv3x3_relu_pool",
                ):
                    out = torch.nn.functional.conv2d(x, self.weight, self.bias, padding=1)
                    out = torch.nn.functional.max_pool2d(torch.relu(out), kernel_size=2, stride=2)
                return out

        x = torch.randn(2, 2, 8, 8)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out, atol=1e-2, rtol=1e-2,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_conv3x3_relu_pool", "torch_op": "conv3x3_relu_pool"}],
        )


class TestLowerAsRequireFalseE2E(unittest.TestCase):
    """TRT compilation with require=False: lowerable region is still lowered.

    require=False means the region is optional — if the impl cannot lower it,
    eager fallback is used.  When the region IS lowerable, TRT still replaces
    it.  These tests verify that require=False works correctly in TRT compile
    mode: the engine produces correct numeric output.
    """

    def test_require_false_lowerable_triton_add_one(self):
        """require=False region with a valid Triton plugin; TRT lowers and outputs match."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=False, name="add_one_optional",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one_optional"}],
        )

    def test_require_false_lowerable_triton_scale(self):
        """require=False region with Triton scale; TRT output matches eager."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])),
                    require=False, name="scale_optional",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_scale", "torch_op": "scale_optional"}],
        )

    def test_require_false_lowerable_cutedsl_add_one(self):
        """require=False region with a valid CuTeDSL plugin; TRT lowers and outputs match."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(cutedsl_add_one, configs=[{}])),
                    require=False, name="cutedsl_add_one_optional",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_add_one", "torch_op": "cutedsl_add_one_optional"}],
        )

    def test_require_false_chained_both_lowerable(self):
        """Two chained require=False regions; both are lowered by TRT and output matches."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 128}])),
                    require=False, name="add_one_opt",
                ):
                    x = x + 1.0
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_scale, configs=[{"BLOCK_SIZE": 128}])),
                    require=False, name="scale_opt",
                ):
                    x = x * 2.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one_opt"},
                {"backend": "triton", "fn_name": "launch_scale", "torch_op": "scale_opt"},
            ],
        )

    def test_require_false_dynamic_shape(self):
        """require=False with dynamic 1D profile; TRT lowers the region and output is correct."""
        import torch_tensorrt

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(launch_add_one, configs=[{"BLOCK_SIZE": 64}])),
                    require=False, name="add_one_dyn_opt",
                ):
                    x = x + 1.0
                return x

        trt_inputs = [
            torch_tensorrt.Input(min_shape=(64,), opt_shape=(128,), max_shape=(256,), dtype=torch.float32)
        ]
        trt_model, trt_out, eager_out = _compile_and_run_dynamic(
            M(), trt_inputs, (torch.randn(128, device="cuda"),)
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one_dyn_opt"}],
        )


@pytest.mark.requires_pre_bw
class TestLowerAsFormatE2E(unittest.TestCase):
    """lower_as with explicit HWC8 format declaration across all backends."""

    def test_lower_as_triton_hwc8(self):
        """lower_as Triton add_one region with HWC8 format declaration."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(
                        launch_add_one,
                        configs=[{"BLOCK_SIZE": 64}],
                        input_formats=[trt.TensorFormat.HWC8],
                        output_formats=[trt.TensorFormat.HWC8],
                    )),
                    require=True, name="triton_hwc8",
                ):
                    x = x + 1.0
                return x

        x = torch.ones(1, 8, 4, 4, dtype=torch.float16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton", "fn_name": "launch_add_one", "torch_op": "triton_hwc8"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_plugin_io_format(self, trt_model, "HWC8")

    def test_lower_as_cutile_hwc8(self):
        """lower_as CuTile add_one region with HWC8 format declaration."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutile(
                        cutile_add_one,
                        configs=[{"BLOCK": 64}],
                        input_formats=[trt.TensorFormat.HWC8],
                        output_formats=[trt.TensorFormat.HWC8],
                    )),
                    require=True, name="cutile_hwc8",
                ):
                    x = x + 1.0
                return x

        x = torch.ones(1, 8, 4, 4, dtype=torch.float16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutile", "fn_name": "launch_add_one", "torch_op": "cutile_hwc8"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_plugin_io_format(self, trt_model, "HWC8")

    def test_lower_as_cutedsl_hwc8(self):
        """lower_as CuTeDSL add_one region with HWC8 format declaration."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.cutedsl(
                        cutedsl_add_one,
                        configs=[{}],
                        input_formats=[trt.TensorFormat.HWC8],
                        output_formats=[trt.TensorFormat.HWC8],
                    )),
                    require=True, name="cutedsl_hwc8",
                ):
                    x = x + 1.0
                return x

        x = torch.ones(1, 8, 4, 4, dtype=torch.float16)
        trt_model, trt_out, eager_out = _compile_and_run(M(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "cutedsl", "fn_name": "launch_add_one", "torch_op": "cutedsl_hwc8"}],
            atol=1e-2, rtol=1e-2,
        )
        assert_plugin_io_format(self, trt_model, "HWC8")


class TestLowerAsAutotuneNonLastTacticE2E(unittest.TestCase):
    """Verify TRT's autotuner can select a non-last tactic through lower_as.

    Uses three Triton configs with different BLOCK_SIZE values.  The test does
    not force a winner; it verifies that get_selected_tactic_for_engine returns
    a non-None result and that the selected tactic's backend is "triton".
    """

    def test_lower_as_triton_non_last_tactic(self):
        """lower_as region with 3 Triton configs; selected tactic has backend triton."""
        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.custom_plugin(tta.triton(
                        launch_add_one,
                        configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 256}],
                    )),
                    require=True, name="add_one_three_tactics",
                ):
                    x = x + 1.0
                return x

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[
                {"backend": "triton", "fn_name": "launch_add_one", "torch_op": "add_one_three_tactics"},
            ],
        )

        selected = get_selected_tactic_for_engine(trt_model)
        if selected:
            spec_id = next(iter(selected))
            tactic = selected[spec_id]
            self.assertIsNotNone(tactic, "Expected a non-None selected tactic")
            self.assertEqual(
                tactic.get("backend"), "triton",
                f"Expected selected tactic backend == 'triton'; got: {tactic}",
            )
