"""IO compatibility between region boundary and impl descriptor for lower_as."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .region_discovery import BoundaryTensor, RegionIO


def _rank_and_dtype(tm: Any) -> Tuple[Optional[int], Optional[torch.dtype]]:
    """Extract rank and dtype from a tensor metadata object.

    Args:
      tm: Tensor metadata object (e.g. ``FakeTensor``, ``TensorMetadata``,
          or ``None``).  Must have ``shape`` and ``dtype`` attributes if
          non-None.

    Returns:
      ``(rank, dtype)`` where ``rank`` is ``len(tm.shape)`` and ``dtype``
      is ``tm.dtype``.  Either or both may be ``None`` if the information
      is unavailable.
    """
    if tm is None:
        return None, None
    shape = getattr(tm, "shape", None)
    dtype = getattr(tm, "dtype", None)
    rank = len(shape) if shape is not None else None
    return rank, dtype


def _get_impl_arity(impl_desc: Any) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(num_inputs, num_outputs)`` if knowable from ``impl_desc``.

    Returns integers only when the spec definitively declares arity (e.g. via
    ``num_inputs`` / ``num_outputs`` attributes or ``io_signatures``).  When
    the spec carries no arity information both values are ``None`` so the
    caller can skip the count check.

    Args:
      impl_desc: An impl descriptor object (e.g. a lower_as impl spec).

    Returns:
      ``(num_inputs, num_outputs)``; either value may be ``None``.
    """
    # Direct attributes take precedence (used by hand-crafted test mocks).
    ni = getattr(impl_desc, "num_inputs", None)
    no = getattr(impl_desc, "num_outputs", None)
    if ni is not None or no is not None:
        return ni, no

    # If io_signatures is present and non-empty, use the first signature's counts
    # (all signatures in a valid list must agree on arity per the spec design).
    sigs_fn = getattr(impl_desc, "io_signatures", None)
    if callable(sigs_fn):
        sigs = sigs_fn()
        if sigs:
            first = sigs[0]
            ni = getattr(first, "num_inputs", None)
            no = getattr(first, "num_outputs", None)
            return ni, no

    return None, None


def _check_slot_compatibility(
    tensors: List[BoundaryTensor],
    sig: Any,
    is_input: bool,
) -> Tuple[bool, Optional[str]]:
    """Check dtype/rank per slot against a single IO signature.

    Args:
      tensors:  List of BoundaryTensor objects for one side (inputs or outputs).
      sig:      A single IO signature object with optional ``required_input_ranks``,
                ``allowed_input_dtypes``, ``required_output_ranks``, and
                ``allowed_output_dtypes`` attributes.
      is_input: True if checking input slots; False for output slots.

    Returns:
      ``(ok, reason)`` where ``ok`` is True on success and ``reason`` is a
      human-readable mismatch description (or ``None``) on failure.
    """
    side = "input" if is_input else "output"
    if is_input:
        req_ranks = getattr(sig, "required_input_ranks", None)
        allowed_dtypes = getattr(sig, "allowed_input_dtypes", None)
    else:
        req_ranks = getattr(sig, "required_output_ranks", None)
        allowed_dtypes = getattr(sig, "allowed_output_dtypes", None)

    for i, bt in enumerate(tensors):
        r, dt = _rank_and_dtype(bt.tensor_meta)
        if req_ranks is not None:
            if i >= len(req_ranks):
                return False, (
                    f"{side} slot {i}: impl has no required_rank entry at index {i} "
                    f"(impl declares {len(req_ranks)} {side}(s))"
                )
            req_r = req_ranks[i]
            if req_r is not None and r is not None and r != req_r:
                return False, (
                    f"{side} slot {i}: rank mismatch "
                    f"(region tensor rank={r}, impl required_rank={req_r})"
                )
        if allowed_dtypes is not None:
            if i >= len(allowed_dtypes):
                return False, (
                    f"{side} slot {i}: impl has no allowed_dtypes entry at index {i} "
                    f"(impl declares {len(allowed_dtypes)} {side}(s))"
                )
            allowed = allowed_dtypes[i]
            if dt is not None and allowed and dt not in allowed:
                return False, (
                    f"{side} slot {i}: dtype mismatch "
                    f"(region tensor dtype={dt}, impl allowed_dtypes={list(allowed)})"
                )
    return True, None


def io_compatible(region_io: RegionIO, impl_desc: Any) -> Tuple[bool, str]:
    """Check IO compatibility between a region boundary and an impl descriptor.

    Performs two levels of checking:
      1. Arity check: input and output counts match what ``impl_desc`` declares.
      2. Per-slot dtype/rank check against each IO signature returned by
         ``impl_desc.io_signatures()`` (if present).  A match against *any*
         signature is sufficient.

    Args:
      region_io: The computed boundary IO for the region (from
                 ``compute_region_io``).
      impl_desc: The impl descriptor to check against (a lower_as impl spec
                 or similar object with optional ``num_inputs``,
                 ``num_outputs``, and ``io_signatures`` attributes).

    Returns:
      ``(is_compatible, reason)`` — ``reason`` is an empty string when
      compatible; a human-readable explanation otherwise.
    """
    ni_region = len(region_io.inputs)
    no_region = len(region_io.outputs)

    # --- Always-on count check ---
    exp_ni, exp_no = _get_impl_arity(impl_desc)
    if exp_ni is not None and exp_ni != ni_region:
        return False, (
            f"input count mismatch: region={ni_region}, impl={exp_ni} "
            f"(region has {ni_region} input(s), impl expects {exp_ni})"
        )
    if exp_no is not None and exp_no != no_region:
        return False, (
            f"output count mismatch: region={no_region}, impl={exp_no} "
            f"(region has {no_region} output(s), impl expects {exp_no})"
        )

    # --- io_signatures check ---
    sigs_fn = getattr(impl_desc, "io_signatures", None)
    if not callable(sigs_fn):
        return True, ""
    sigs = sigs_fn()
    if not sigs:
        return True, ""

    last_reason = "no IO signature matched the region boundary"
    for sig in sigs:
        num_in = getattr(sig, "num_inputs", None)
        num_out = getattr(sig, "num_outputs", None)
        if num_in is not None and num_in != ni_region:
            last_reason = (
                f"input count mismatch: region has {ni_region} input(s) "
                f"(region={ni_region}), signature expects {num_in}"
            )
            continue
        if num_out is not None and num_out != no_region:
            last_reason = (
                f"output count mismatch: region has {no_region} output(s) "
                f"(region={no_region}), signature expects {num_out}"
            )
            continue

        ok_in, r_in = _check_slot_compatibility(region_io.inputs, sig, is_input=True)
        if not ok_in:
            last_reason = r_in
            continue
        ok_out, r_out = _check_slot_compatibility(region_io.outputs, sig, is_input=False)
        if not ok_out:
            last_reason = r_out
            continue

        # All checks passed for this signature.
        return True, ""

    return False, last_reason
