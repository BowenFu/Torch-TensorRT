"""Argument validation for TTA annotation constraints.

Validates calibration inputs (real vs. fake tensors), autocast mode/require
combinations, and quantize mode strings before they are consumed by the
lowering pipeline.

All public functions in this module raise :class:`ValueError` for invalid
inputs and are silent (return ``None``) for valid ones.  This makes them
suitable for use as guard clauses at the start of API entry-points.
"""

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _is_fake_tensor(x: Any) -> bool:
    """Return ``True`` if *x* is a :class:`torch.fx.experimental.proxy_tensor.FakeTensor`.

    A fake tensor carries shape and dtype metadata but no real data; passing
    one to a calibration routine would produce meaningless quantization scales.

    Detection is duck-typed via the ``is_fake`` attribute rather than a
    direct ``isinstance`` check to avoid importing ``torch.fx`` at module
    load time and to remain compatible with any torch-internal fake-tensor
    subclass.

    Args:
        x: Any Python object to inspect.

    Returns:
        ``True`` if *x* looks like a fake tensor, ``False`` otherwise.
    """
    if not hasattr(x, "is_fake"):
        return False
    return bool(getattr(x, "is_fake", False))


def validate_calib_inputs_for_quantize(calib_inputs: Sequence[Any]) -> None:
    """Raise if any tensor in *calib_inputs* is a fake tensor.

    ``tta.quantize`` requires real, representative input data for
    calibration so that per-channel or per-tensor quantization scales are
    computed from the true value distribution.  Fake tensors carry only
    shape/dtype information; using them would produce incorrect (and
    effectively random) scales, leading to silent accuracy degradation.

    Valid input:
        A sequence of real ``torch.Tensor`` objects (CPU or CUDA) that are
        representative of the model's expected inputs.

    Invalid input:
        Any element that is a fake tensor (i.e. ``x.is_fake`` is truthy),
        including those produced by ``torch.export.export`` tracing or
        ``torch.compile`` symbolic execution.

    Args:
        calib_inputs: Sequence of inputs to inspect.  Non-tensor elements
            are silently accepted.

    Raises:
        ValueError: If any ``torch.Tensor`` element in *calib_inputs* is a
            fake tensor.  The error message identifies the offending index
            and explains how to supply real tensors instead.
    """
    import torch

    for i, x in enumerate(calib_inputs):
        if isinstance(x, torch.Tensor) and _is_fake_tensor(x):
            raise ValueError(
                f"tta.quantize requires real input data for calibration so that "
                f"quantization scales reflect the true value distribution. "
                f"Input at index {i} is a fake tensor. "
                f"Pass real, representative tensors via tta.compile(..., inputs=...)."
            )


def validate_autocast_args(*, mode: str, require: bool) -> None:
    """Validate arguments for ``tta.autocast``.

    Valid combinations:

    - ``mode="fp16", require=False`` — request FP16 autocast, fall back
      silently if unsupported.
    - ``mode="fp16", require=True`` — require FP16 autocast; raise at
      compile time if the hardware or TRT version does not support it.
    - ``mode="bf16", require=False / True`` — same semantics for BF16.
    - ``mode="disable", require=False`` — explicitly disable autocast for
      this region.

    Invalid combinations:

    - ``mode="disable", require=True`` — logically contradictory: one
      cannot *require* that autocast be disabled; TTA raises immediately.
    - Any ``mode`` not in ``("fp16", "bf16", "disable")`` — unsupported.

    Args:
        mode: Autocast precision mode string.  Must be one of
            ``"fp16"``, ``"bf16"``, or ``"disable"``.
        require: If ``True``, TTA treats failure to lower this region as a
            hard error rather than a silent fallback.  Incompatible with
            ``mode="disable"``.

    Raises:
        ValueError: If *mode* is not a recognised string, or if
            ``mode="disable"`` is combined with ``require=True``.
    """
    if mode not in ("fp16", "bf16", "disable"):
        raise ValueError(
            f"[tta.autocast] unsupported mode {mode!r}. "
            f"Supported modes are: 'fp16', 'bf16', 'disable'."
        )
    if mode == "disable" and require:
        raise ValueError(
            "[tta.autocast] mode='disable' is incompatible with require=True. "
            "Disabling autocast is a no-op lowering; it cannot be 'required'."
        )


def validate_quantize_args(*, mode: str, require: bool) -> None:
    """Validate arguments for ``tta.quantize``.

    Supported modes:

    - ``"auto"``   — let TTA choose the best available quantization format
      for the target GPU (e.g. NVFP4 on Blackwell, INT8 elsewhere).
    - ``"int8"``   — explicit INT8 (symmetric per-tensor or per-channel).
    - ``"fp8"``    — explicit FP8 (E4M3 format via ModelOpt).
    - ``"nvfp4"``  — explicit NVFP4 (Blackwell micro-scaling format).

    MXFP8 and MXFP4 are intentionally absent from this list even though
    ModelOpt exposes presets for them.  Direct Blackwell tests showed that
    torch-tensorrt's current ``dynamic_block_quantize`` conversion path
    fails during TRT lowering for those formats, while NVFP4 succeeds.
    They will be added here once the underlying conversion path is fixed.

    The *require* argument follows the same semantics as in ``tta.autocast``
    and ``tta.export_as``: when ``True``, a failure to lower the annotated
    region as a quantized sub-graph raises immediately instead of falling
    back to FP32.

    Args:
        mode: Quantization mode string.  Must be one of
            ``"auto"``, ``"int8"``, ``"fp8"``, or ``"nvfp4"``.
        require: If ``True``, failure to quantize the region is a hard
            error.  Has no effect on which modes are valid.

    Raises:
        ValueError: If *mode* is not a non-empty string in the supported
            set.  The error message lists all supported modes.
    """
    supported = ("auto", "int8", "fp8", "nvfp4")
    if not isinstance(mode, str) or not mode or mode not in supported:
        raise ValueError(
            f"[tta.quantize] Unsupported mode {mode!r}. "
            f"Supported modes are: {', '.join(repr(m) for m in supported)}."
        )
