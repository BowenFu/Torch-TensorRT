"""Backend launch recorders for Triton, cuTILE, and CuTe DSL.

These recorder classes are used exclusively inside the *autotune sandbox* —
the dry-run environment in which the TTA autotune pass executes a user's
``launch_fn`` without actually dispatching GPU work.  The sandbox replaces
real kernel objects with recorder proxies so that launch parameters (grid,
block, args) can be captured and later used to build TRT QDP plugin
descriptors, without touching the GPU or requiring CUDA to be initialised.

Each recorder mirrors the call protocol of its target backend:

Backend         | Call protocol                              | Recorder class
----------------|--------------------------------------------|--------------------
Triton          | ``kernel[grid](*args, **kwargs)``          | TritonLaunchRecorder
cuTILE          | ``prog(*args, **kwargs)``                  | CuTileLaunchRecorder
CuTe DSL (run)  | ``compiled.run(*args, **kwargs)``          | CuTeRunRecorder
CuTe DSL (kernel)| ``kernel(*args)(...).launch(grid, block)``| CuTeDSLKernelRecorder

After the sandbox ``launch_fn`` returns, the autotune pass inspects the
populated fields on the recorder instance to retrieve the captured
parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class TritonLaunchRecorder:
    """Proxy that mimics a Triton kernel, recording launches instead of running.

    Used in the autotune sandbox to capture the parameters that a user's
    ``launch_fn`` passes to a Triton kernel without executing any GPU work.

    The Triton call protocol is ``kernel[grid](*args, **kwargs)``.  This
    recorder implements ``__getitem__`` to capture the grid and returns a
    closure that captures the positional and keyword arguments when called.

    Attributes:
        real_kernel: The original Triton kernel object being proxied.  Not
            invoked by the recorder; held for reference only (e.g. to read
            kernel metadata outside the sandbox).
        grid: Grid tuple captured from ``kernel[grid]``.  ``None`` until
            a launch is recorded.
        args: Positional arguments captured from the launcher call.  ``None``
            until a launch is recorded.
        kwargs: Keyword arguments captured from the launcher call.  ``None``
            until a launch is recorded.
    """

    real_kernel: Any
    grid: Optional[Any] = None
    args: Optional[Tuple[Any, ...]] = None
    kwargs: Optional[Dict[str, Any]] = None

    def __getitem__(self, grid: Any) -> Any:
        """Capture ``kernel[grid]`` and return a launcher closure.

        Args:
            grid: The grid value (e.g. a tuple or a lambda) passed inside
                ``kernel[grid]``.

        Returns:
            A callable that, when called with ``(*args, **kwargs)``, stores
            those values on this recorder.
        """

        def _launcher(*args: Any, **kwargs: Any) -> None:
            self.grid = grid
            self.args = args
            self.kwargs = kwargs

        return _launcher


@dataclass
class CuTileLaunchRecorder:
    """Proxy that mimics a cuTILE program object, recording calls instead of running.

    Used in the autotune sandbox to capture the arguments passed to a cuTILE
    compiled program without executing any GPU work.

    The cuTILE call protocol is ``prog(*args, **kwargs)`` where *prog* is an
    already-compiled ``cuda.tile`` program object.  This recorder implements
    ``__call__`` to capture positional and keyword arguments.

    Note: cuTILE programs do not expose a separate ``[grid]`` subscript; the
    grid is typically embedded in the program configuration or passed as a
    keyword argument.  The ``grid`` field is reserved for future use when
    cuTILE exposes per-launch grid control.

    Attributes:
        real_prog: The original cuTILE program object being proxied.  Not
            invoked by the recorder; held for reference only.
        args: Positional arguments captured from the call.  ``None`` until
            a launch is recorded.
        kwargs: Keyword arguments captured from the call.  ``None`` until
            a launch is recorded.
        grid: Reserved for future cuTILE grid capture.  Always ``None``
            in the current implementation.
    """

    real_prog: Any
    args: Optional[Tuple[Any, ...]] = None
    kwargs: Optional[Dict[str, Any]] = None
    grid: Optional[Tuple[int, ...]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Capture ``prog(*args, **kwargs)`` without running the program.

        Args:
            *args: Positional arguments forwarded to the cuTILE program.
            **kwargs: Keyword arguments forwarded to the cuTILE program.
        """
        self.args = args
        self.kwargs = kwargs


@dataclass
class CuTeRunRecorder:
    """Proxy that records CuTe DSL ``compiled.run(...)`` calls.

    Used in the autotune sandbox when a user's ``launch_fn`` calls
    ``compiled.run(...)`` on a CuTe DSL compiled object.  The recorder
    captures the ``compiled`` object and arguments so the autotune pass can
    inspect them after the sandbox returns.

    Unlike ``CuTeDSLKernelRecorder`` (which wraps a ``@cute.kernel``
    decorator result), this recorder targets the lower-level
    ``compiled.run`` API available on ``CuTeDSL`` compiled artifacts.

    Attributes:
        compiled: The CuTe DSL compiled object passed to ``record_run``.
            ``None`` until ``record_run`` is called.
        args: Positional arguments captured from ``compiled.run``.  ``None``
            until ``record_run`` is called.
        kwargs: Keyword arguments captured from ``compiled.run``.  ``None``
            until ``record_run`` is called.
    """

    compiled: Any = None
    args: Optional[Tuple[Any, ...]] = None
    kwargs: Optional[Dict[str, Any]] = None

    def record_run(self, compiled: Any, *args: Any, **kwargs: Any) -> None:
        """Record a ``compiled.run(...)`` call.

        Args:
            compiled: The CuTe DSL compiled object whose ``run`` method was
                called.
            *args: Positional arguments forwarded to ``compiled.run``.
            **kwargs: Keyword arguments forwarded to ``compiled.run``.
        """
        self.compiled = compiled
        self.args = args
        self.kwargs = kwargs


class _CuTeDSLLaunchProxy:
    """Intermediate object returned by ``CuTeDSLKernelRecorder.__call__``.

    Represents the result of calling a ``@cute.kernel`` object (the
    compiled artifact) and exposes a ``.launch(grid, block)`` method that
    the autotune sandbox intercepts to record grid and block dimensions.

    This class is an implementation detail of ``CuTeDSLKernelRecorder`` and
    is not intended to be instantiated directly.
    """

    def __init__(self, recorder: "CuTeDSLKernelRecorder") -> None:
        self._recorder = recorder

    def launch(
        self,
        grid: Any = (1, 1, 1),
        block: Any = (1, 1, 1),
        **kwargs: Any,
    ) -> None:
        """Capture the grid and block dimensions from a ``.launch()`` call.

        Args:
            grid: Grid dimensions tuple (x, y, z).  Non-tuple values are
                coerced to a tuple.
            block: Block dimensions tuple (x, y, z).  Non-tuple values are
                coerced to a tuple.
            **kwargs: Additional keyword arguments are accepted but ignored;
                they are not recorded because they are not needed for TRT
                QDP descriptor construction.
        """
        grid_t = tuple(grid) if not isinstance(grid, tuple) else grid
        block_t = tuple(block) if not isinstance(block, tuple) else block
        if len(grid_t) != 3:
            raise ValueError(
                f"_CuTeDSLLaunchProxy.launch: grid must have exactly 3 elements (x, y, z), "
                f"got {len(grid_t)}: {grid_t!r}"
            )
        if len(block_t) != 3:
            raise ValueError(
                f"_CuTeDSLLaunchProxy.launch: block must have exactly 3 elements (x, y, z), "
                f"got {len(block_t)}: {block_t!r}"
            )
        self._recorder.grid = grid_t
        self._recorder.block = block_t


@dataclass
class CuTeDSLKernelRecorder:
    """Proxy for ``@cute.kernel`` objects that records grid/block on ``.launch()``.

    Used in the autotune sandbox to capture the launch configuration
    (grid and block dimensions) of a CuTe DSL kernel without executing GPU
    work.

    The CuTe DSL call protocol for ``@cute.kernel`` decorated functions is::

        result = kernel(*args, **kwargs)   # compile / specialize
        result.launch(grid=..., block=...) # dispatch

    This recorder implements ``__call__`` to return a ``_CuTeDSLLaunchProxy``
    that records the subsequent ``.launch()`` call, storing ``grid`` and
    ``block`` on this instance.

    Attributes:
        real_kernel: The original ``@cute.kernel`` object being proxied.
            Not invoked by the recorder; held for reference only.
        grid: Grid dimensions captured from ``.launch(grid=...)``.  ``None``
            until a launch is recorded.
        block: Block dimensions captured from ``.launch(block=...)``.
            ``None`` until a launch is recorded.
    """

    real_kernel: Any
    grid: Optional[Tuple[Any, ...]] = None
    block: Optional[Tuple[Any, ...]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> _CuTeDSLLaunchProxy:
        """Return a launch proxy that will record grid/block on ``.launch()``.

        Args:
            *args: Positional arguments passed to the kernel (ignored;
                not needed for TRT descriptor construction).
            **kwargs: Keyword arguments passed to the kernel (ignored).

        Returns:
            A ``_CuTeDSLLaunchProxy`` bound to this recorder.
        """
        return _CuTeDSLLaunchProxy(self)
