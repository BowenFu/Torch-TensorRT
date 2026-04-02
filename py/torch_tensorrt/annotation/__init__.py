"""
Torch-TensorRT Annotation Layer (TTA) — custom_plugin API.

Provides descriptor types and factory functions for defining custom TensorRT
AOT QDP plugins backed by Triton, CuTile, or CuTeDSL kernels.

Usage::

    import torch_tensorrt.annotation as tta

    # Triton kernel descriptor
    spec = tta.custom_plugin(tta.triton(my_triton_kernel, configs=[{"BLOCK_SIZE": 128}]))

    # CuTile kernel descriptor
    spec = tta.custom_plugin(tta.cutile(my_cutile_kernel, arch=120))

    # CuTeDSL kernel descriptor
    spec = tta.custom_plugin(tta.cutedsl(my_cutedsl_kernel))
"""

import logging

from ._errors import TTADiagnosticError

from ._specs import (
    AnnotationMetadata,
    KernelImplSpec,
    CuTeDSLSpec,
    CuTileSpec,
    TritonSpec,
    cutedsl,
    cutile,
    triton,
    normalize_impl_to_spec,
)

from ._custom_plugin._descriptor import CustomPluginSpec, custom_plugin

_logger = logging.getLogger(__name__)

__all__ = [
    # Error types
    "TTADiagnosticError",
    # Descriptor types
    "AnnotationMetadata",
    "TritonSpec",
    "CuTileSpec",
    "CuTeDSLSpec",
    "CustomPluginSpec",
    # Factory functions
    "custom_plugin",
    "triton",
    "cutile",
    "cutedsl",
]
