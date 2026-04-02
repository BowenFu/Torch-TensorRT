"""E2E tests for trt_plugins.custom_op(impl=tta.custom_plugin(...)).

Verifies that a ``tta.CustomPluginSpec`` passed as ``impl=`` to
``torch_tensorrt.dynamo.conversion.plugins.custom_op`` correctly:
  1. Auto-registers torch.library.custom_op + register_fake — no boilerplate.
  2. Registers the QDP plugin via TTA's AOT backend (Triton).
  3. Compiles to a single TRT engine via torch_tensorrt.compile().
  4. Produces numerically accurate results vs. PyTorch eager.
"""

import unittest

import torch
import torch.nn as nn
import torch_tensorrt
import torch_tensorrt.annotation as tta
import torch_tensorrt.dynamo.conversion.plugins as trt_plugins
import triton
import triton.language as tl

from ._e2e_common import _compile_and_run, assert_trt_compiled


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _add_one_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + 1.0, mask=mask)


def _launch_add_one(x, out, BLOCK_SIZE=128):
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _add_one_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


@triton.jit
def _add_two_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _launch_add_two(x, y, out, BLOCK_SIZE=128):
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _add_two_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Register via trt_plugins.custom_op(impl=tta.custom_plugin(...))
# No @torch.library boilerplate needed — auto_register_torch_op handles it.
# ---------------------------------------------------------------------------

trt_plugins.custom_op(
    "torchtrt_trt_plugins_e2e::add_one",
    impl=tta.custom_plugin(
        tta.triton(_launch_add_one, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)

trt_plugins.custom_op(
    "torchtrt_trt_plugins_e2e::add_two",
    impl=tta.custom_plugin(
        tta.triton(_launch_add_two, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x, y: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrtPluginsCustomOpE2E(unittest.TestCase):
    """E2E: custom_op(impl=tta.custom_plugin(...)) compiles and runs correctly."""

    def test_add_one_single_input(self):
        """Single-input plugin: output = x + 1.0"""

        class M(nn.Module):
            def forward(self, x):
                return torch.ops.torchtrt_trt_plugins_e2e.add_one.default(x)

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(128),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_add_two_two_inputs(self):
        """Two-input plugin: output = x + y"""

        class M(nn.Module):
            def forward(self, x, y):
                return torch.ops.torchtrt_trt_plugins_e2e.add_two.default(x, y)

        trt_model, trt_out, eager_out = _compile_and_run(
            M(), (torch.randn(256), torch.randn(256))
        )
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )

    def test_add_one_in_larger_graph(self):
        """Plugin fused inside a larger graph with aten ops."""

        class M(nn.Module):
            def forward(self, x):
                x = x * 2.0
                x = torch.ops.torchtrt_trt_plugins_e2e.add_one.default(x)
                return x + 0.5

        trt_model, trt_out, eager_out = _compile_and_run(M(), (torch.randn(256),))
        assert_trt_compiled(
            self, trt_model, trt_out, eager_out,
            expected_tta_metadata=[{"backend": "triton"}],
        )


if __name__ == "__main__":
    unittest.main()
