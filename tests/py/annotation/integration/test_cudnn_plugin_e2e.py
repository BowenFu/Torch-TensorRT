"""
cuDNN-dependent plugin tests (InstanceNormalization_TRT with self_attr weights).

Requires cuDNN 8 and a cc < 10 GPU.  Runs in the cuDNN child process
spawned by conftest.py — never collected in the Blackwell main process.
"""

import unittest

import tensorrt as trt
import torch
import torch.nn as nn

import torch_tensorrt.annotation as tta

from ._e2e_common import (
    _compile_and_run,
    _has_cudnn8,
    assert_trt_compiled,
    get_layers_with_tta_metadata,
    tta_compile,
)

C = 8


def _make_instnorm_model(eps=1e-5):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(C))
            self.bias_param = nn.Parameter(torch.zeros(C))

        @tta.export_as(
            impl=tta.plugin(
                "InstanceNormalization_TRT", "3", "",
                epsilon=eps,
                scales=tta.self_attr("weight"),
                bias=tta.self_attr("bias_param"),
            )
        )
        def instnorm(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.instance_norm(
                x, self.weight, self.bias_param, eps=eps,
            )

        def forward(self, x):
            return self.instnorm(x)

    return M()


class TestCuDNNPluginSelfAttr(unittest.TestCase):
    """self_attr weights with InstanceNormalization_TRT."""

    @classmethod
    def setUpClass(cls):
        assert _has_cudnn8(), "cuDNN not available"
        trt.init_libnvinfer_plugins(None, "")

    def test_plugin_self_attr_weights(self):
        x = torch.randn(2, C, 4, 4)
        trt_model, trt_out, eager_out = _compile_and_run(_make_instnorm_model(), (x,))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )

    def test_plugin_self_attr_weights_has_tta_metadata(self):
        model = _make_instnorm_model().eval().cuda()
        x = torch.randn(2, C, 4, 4).cuda()
        with torch.no_grad():
            eager_out = model(x)
        trt_model = tta_compile(
            model,
            inputs=(x,),
        )
        with torch.no_grad():
            trt_out = trt_model(x)
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "plugin", "plugin_name": "InstanceNormalization_TRT"}],
        )
