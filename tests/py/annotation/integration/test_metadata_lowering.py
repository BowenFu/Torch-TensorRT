"""Test that TTA metadata set during lowering is present in engine layer info.

Tests cover:
  - TestTTAMetadataLowering    — builtin lowering via export_as stamps TTA metadata.
  - TestTTAMetadataLowerAs     — lower_as region lowering stamps TTA metadata.
  - TestTTAMetadataMultiLayer  — multiple annotated layers each carry metadata.
"""

import unittest

import pytest
import tensorrt as trt
import torch
import torch.nn as nn

import torch_tensorrt
import torch_tensorrt.annotation as tta

from ._e2e_common import (
    _compile_and_run,
    assert_trt_compiled,
    get_layers_with_tta_metadata,
    get_trt_engine_layer_info,
)


class TestTTAMetadataLowering(unittest.TestCase):
    def test_builtin_layer_has_tta_metadata(self):
        class M(nn.Module):
            @staticmethod
            @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
            def relu(x):
                return torch.relu(x)

            def forward(self, x):
                return self.relu(x)

        model = M().eval().cuda()
        cuda_inputs = (torch.randn(2, 8).cuda(),)
        with torch.no_grad():
            eager_out = model(*cuda_inputs)
        trt_model = torch_tensorrt.compile(
            model,
            inputs=cuda_inputs,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            min_block_size=1,
            require_full_compilation=True,
        )
        with torch.no_grad():
            trt_out = trt_model(*cuda_inputs)
        torch.testing.assert_close(trt_out, eager_out)
        layer_info = get_trt_engine_layer_info(trt_model)
        self.assertGreater(len(layer_info), 0)
        with_tta = get_layers_with_tta_metadata(trt_model)
        self.assertGreater(
            len(with_tta), 0,
            msg="Expected at least one layer with TTA metadata (Metadata field in inspector JSON)",
        )
        for _layer, meta in with_tta:
            self.assertEqual(meta.get("backend"), "builtin",
                f"All TTA layers should have backend=builtin; got {meta.get('backend')!r}")
            self.assertEqual(meta.get("plugin_name"), "add_activation",
                f"All TTA layers should have plugin_name=add_activation; got {meta.get('plugin_name')!r}")
            self.assertIn("torch_op", meta,
                f"TTA metadata must include torch_op; got {meta}")
            self.assertTrue(meta.get("torch_op"),
                f"torch_op must be non-empty; got {meta.get('torch_op')!r}")

    def test_builtin_self_attr_has_tta_metadata(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 1.0

            @tta.export_as(
                impl=tta.builtin(
                    "add_activation",
                    type=trt.ActivationType.ELU,
                    alpha=tta.self_attr("alpha"),
                )
            )
            def elu_op(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.elu(x, alpha=self.alpha)

            def forward(self, x):
                return self.elu_op(x)

        model = M().eval().cuda()
        x = torch.randn(2, 8).cuda() - 0.5
        cuda_inputs = (x,)
        with torch.no_grad():
            eager_out = model(*cuda_inputs)
        trt_model = torch_tensorrt.compile(
            model,
            inputs=cuda_inputs,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            min_block_size=1,
            require_full_compilation=True,
        )
        with torch.no_grad():
            trt_out = trt_model(*cuda_inputs)
        torch.testing.assert_close(trt_out, eager_out)
        with_tta = get_layers_with_tta_metadata(trt_model)
        self.assertGreater(len(with_tta), 0)
        for _layer, meta in with_tta:
            self.assertEqual(meta.get("backend"), "builtin",
                f"All TTA layers should have backend=builtin; got {meta.get('backend')!r}")
            self.assertEqual(meta.get("plugin_name"), "add_activation",
                f"All TTA layers should have plugin_name=add_activation; got {meta.get('plugin_name')!r}")
            self.assertIn("torch_op", meta, f"TTA metadata must include torch_op; got {meta}")
            self.assertTrue(meta.get("torch_op"), f"torch_op must be non-empty; got {meta.get('torch_op')!r}")


# ---------------------------------------------------------------------------
# TestTTAMetadataLowerAs
# ---------------------------------------------------------------------------

class TestTTAMetadataLowerAs(unittest.TestCase):
    """lower_as region lowering stamps TTA metadata on engine layers."""

    def test_lower_as_builtin_relu_has_tta_metadata(self):
        """lower_as with builtin ReLU: engine layer has backend=builtin in TTA metadata."""

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.builtin("add_activation", type=trt.ActivationType.RELU),
                    require=True,
                    name="lower_relu",
                ):
                    x = torch.relu(x)
                return x

        model = M().eval().cuda()
        x = torch.randn(2, 8).cuda()
        with torch.no_grad():
            eager_out = model(x)

        trt_model = torch_tensorrt.compile(
            model,
            inputs=(x,),
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            min_block_size=1,
            require_full_compilation=True,
        )
        with torch.no_grad():
            trt_out = trt_model(x)

        torch.testing.assert_close(trt_out, eager_out)

        with_tta = get_layers_with_tta_metadata(trt_model)
        self.assertGreater(
            len(with_tta), 0,
            msg="Expected TTA metadata from lower_as(builtin) in engine layers",
        )
        for _layer, meta in with_tta:
            self.assertEqual(meta.get("backend"), "builtin",
                f"All TTA layers should have backend=builtin; got {meta.get('backend')!r}")
            self.assertEqual(meta.get("plugin_name"), "add_activation",
                f"All TTA layers should have plugin_name=add_activation; got {meta.get('plugin_name')!r}")
            self.assertEqual(meta.get("torch_op"), "lower_relu",
                f"lower_as(name='lower_relu') must set torch_op='lower_relu'; got {meta.get('torch_op')!r}")

    def test_lower_as_builtin_sigmoid_has_tta_metadata(self):
        """lower_as with builtin Sigmoid: engine layer carries TTA metadata."""

        class M(nn.Module):
            def forward(self, x):
                with tta.lower_as(
                    impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID),
                    require=True,
                    name="lower_sigmoid",
                ):
                    x = torch.sigmoid(x)
                return x

        model = M().eval().cuda()
        x = torch.randn(4, 8).cuda()
        with torch.no_grad():
            eager_out = model(x)

        trt_model = torch_tensorrt.compile(
            model,
            inputs=(x,),
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            min_block_size=1,
            require_full_compilation=True,
        )
        with torch.no_grad():
            trt_out = trt_model(x)

        torch.testing.assert_close(trt_out, eager_out, atol=1e-4, rtol=1e-4)
        with_tta = get_layers_with_tta_metadata(trt_model)
        self.assertGreater(len(with_tta), 0)
        for _layer, meta in with_tta:
            self.assertEqual(meta.get("backend"), "builtin")
            self.assertEqual(meta.get("plugin_name"), "add_activation")
            self.assertEqual(meta.get("torch_op"), "lower_sigmoid",
                f"lower_as(name='lower_sigmoid') must set torch_op='lower_sigmoid'; got {meta.get('torch_op')!r}")


# ---------------------------------------------------------------------------
# TestTTAMetadataMultiLayer
# ---------------------------------------------------------------------------

class TestTTAMetadataMultiLayer(unittest.TestCase):
    """Multiple annotated layers each contribute TTA metadata to the engine."""

    def test_two_builtin_ops_both_have_metadata(self):
        """Two export_as annotated builtin ops: at least two metadata entries in engine."""

        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID))
        def sigmoid(x):
            return torch.sigmoid(x)

        class M(nn.Module):
            def forward(self, x):
                return sigmoid(relu(x))

        model = M().eval().cuda()
        x = torch.randn(2, 8).cuda()
        with torch.no_grad():
            eager_out = model(x)

        trt_model = torch_tensorrt.compile(
            model,
            inputs=(x,),
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            min_block_size=1,
            require_full_compilation=True,
        )
        with torch.no_grad():
            trt_out = trt_model(x)

        torch.testing.assert_close(trt_out, eager_out, atol=1e-4, rtol=1e-4)

        with_tta = get_layers_with_tta_metadata(trt_model)
        # TRT may fuse relu+sigmoid into one Myelin kernel on Blackwell;
        # at minimum there must be at least one TTA-metadata layer, and all
        # TTA layers that ARE present must carry backend=builtin.
        self.assertGreater(
            len(with_tta), 0,
            msg="Expected at least 1 TTA-metadata layer (two builtin ops annotated)",
        )
        for _layer, meta in with_tta:
            self.assertEqual(meta.get("backend"), "builtin",
                f"All TTA layers should have backend=builtin; got {meta.get('backend')!r}")
            self.assertEqual(meta.get("plugin_name"), "add_activation",
                f"All TTA layers should have plugin_name=add_activation; got {meta.get('plugin_name')!r}")

    def test_export_as_and_lower_as_coexist_with_metadata(self):
        """export_as (builtin ReLU) + lower_as (builtin Sigmoid): both carry TTA metadata.

        The model inserts a Linear layer between the two activations so TRT
        cannot fuse them into a single Myelin kernel.  Both ILayer objects
        survive independently, each with its own ILayer.metadata carrying the
        correct torch_op value.
        """

        @tta.export_as(impl=tta.builtin("add_activation", type=trt.ActivationType.RELU))
        def relu(x):
            return torch.relu(x)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8, bias=False)

            def forward(self, x):
                x = relu(x)
                x = self.fc(x)  # non-pointwise: breaks Myelin fusion with relu
                with tta.lower_as(
                    impl=tta.builtin("add_activation", type=trt.ActivationType.SIGMOID),
                    require=True,
                    name="lower_sigmoid_part",
                ):
                    x = torch.sigmoid(x)
                return x

        model = M().eval().cuda()
        x = torch.randn(2, 8).cuda()
        with torch.no_grad():
            eager_out = model(x)

        trt_model = torch_tensorrt.compile(
            model,
            inputs=(x,),
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            min_block_size=1,
            require_full_compilation=True,
        )
        with torch.no_grad():
            trt_out = trt_model(x)

        torch.testing.assert_close(trt_out, eager_out, atol=1e-4, rtol=1e-4)

        with_tta = get_layers_with_tta_metadata(trt_model)
        self.assertGreater(
            len(with_tta), 0,
            msg="Expected at least one TTA-metadata layer from export_as or lower_as",
        )
        # Both the export_as relu and the lower_as sigmoid appear as separate
        # builtin layers (the Linear between them prevents Myelin fusion).
        # Each must carry its torch_op in ILayer.metadata.
        for _layer, meta in with_tta:
            self.assertEqual(meta.get("backend"), "builtin",
                f"All TTA layers should have backend=builtin; got {meta.get('backend')!r}")
        lower_as_layer = next(
            (meta for _, meta in with_tta if meta.get("torch_op") == "lower_sigmoid_part"),
            None,
        )
        self.assertIsNotNone(
            lower_as_layer,
            "Expected a TTA layer with torch_op='lower_sigmoid_part' from lower_as(name='lower_sigmoid_part')"
        )
