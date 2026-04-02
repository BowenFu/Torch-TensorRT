"""Unit tests for TTA launch recorder classes."""

import unittest

from torch_tensorrt.annotation._recorders import (
    CuTeDSLKernelRecorder,
    CuTileLaunchRecorder,
    TritonLaunchRecorder,
    _CuTeDSLLaunchProxy,
)


class TestTritonLaunchRecorder(unittest.TestCase):
    def _make_kernel(self):
        class FakeTritonKernel:
            pass
        return FakeTritonKernel()

    def test_getitem_returns_callable(self):
        rec = TritonLaunchRecorder(real_kernel=self._make_kernel())
        launcher = rec[(4, 1, 1)]
        self.assertTrue(callable(launcher))

    def test_launcher_captures_grid(self):
        rec = TritonLaunchRecorder(real_kernel=self._make_kernel())
        rec[(8, 2, 1)](1, 2, 3, key=42)
        self.assertEqual(rec.grid, (8, 2, 1))

    def test_launcher_captures_args(self):
        rec = TritonLaunchRecorder(real_kernel=self._make_kernel())
        rec[(1,)]("a", "b", c="c")
        self.assertEqual(rec.args, ("a", "b"))

    def test_launcher_captures_kwargs(self):
        rec = TritonLaunchRecorder(real_kernel=self._make_kernel())
        rec[(1,)](x=10, y=20)
        self.assertEqual(rec.kwargs, {"x": 10, "y": 20})

    def test_grid_none_before_launch(self):
        rec = TritonLaunchRecorder(real_kernel=self._make_kernel())
        self.assertIsNone(rec.grid)
        self.assertIsNone(rec.args)
        self.assertIsNone(rec.kwargs)

    def test_real_kernel_not_invoked(self):
        called = []
        class Kernel:
            def __getitem__(self, grid):
                called.append(grid)
                return lambda *a, **kw: None
        rec = TritonLaunchRecorder(real_kernel=Kernel())
        rec[(1,)]()
        self.assertEqual(called, [])  # recorder short-circuits; real kernel not called

    def test_lambda_grid(self):
        rec = TritonLaunchRecorder(real_kernel=self._make_kernel())
        grid_fn = lambda meta: (meta["N"] // 128,)
        rec[grid_fn]()
        self.assertIs(rec.grid, grid_fn)


class TestCuTileLaunchRecorder(unittest.TestCase):
    def _make_prog(self):
        class FakeProg:
            pass
        return FakeProg()

    def test_call_captures_args(self):
        rec = CuTileLaunchRecorder(real_prog=self._make_prog())
        rec(1, 2, 3)
        self.assertEqual(rec.args, (1, 2, 3))

    def test_call_captures_kwargs(self):
        rec = CuTileLaunchRecorder(real_prog=self._make_prog())
        rec(a=1, b=2)
        self.assertEqual(rec.kwargs, {"a": 1, "b": 2})

    def test_args_none_before_call(self):
        rec = CuTileLaunchRecorder(real_prog=self._make_prog())
        self.assertIsNone(rec.args)
        self.assertIsNone(rec.kwargs)

    def test_grid_always_none(self):
        rec = CuTileLaunchRecorder(real_prog=self._make_prog())
        rec(1)
        self.assertIsNone(rec.grid)

    def test_real_prog_not_invoked(self):
        called = []
        class Prog:
            def __call__(self, *a, **kw):
                called.append(a)
        rec = CuTileLaunchRecorder(real_prog=Prog())
        rec(42)
        self.assertEqual(called, [])


class TestCuTeDSLKernelRecorder(unittest.TestCase):
    def _make_kernel(self):
        class FakeCuTeKernel:
            pass
        return FakeCuTeKernel()

    def test_call_returns_proxy(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        proxy = rec(1, 2, 3)
        self.assertIsInstance(proxy, _CuTeDSLLaunchProxy)

    def test_launch_captures_grid(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        proxy = rec(1, 2, 3)
        proxy.launch(grid=(2, 4, 8), block=(128, 1, 1))
        self.assertEqual(rec.grid, (2, 4, 8))

    def test_launch_captures_block(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        rec().launch(grid=(1, 1, 1), block=(256, 1, 1))
        self.assertEqual(rec.block, (256, 1, 1))

    def test_grid_block_none_before_launch(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        self.assertIsNone(rec.grid)
        self.assertIsNone(rec.block)

    def test_launch_coerces_list_to_tuple(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        rec().launch(grid=[4, 2, 1], block=[64, 1, 1])
        self.assertEqual(rec.grid, (4, 2, 1))
        self.assertEqual(rec.block, (64, 1, 1))

    def test_launch_rejects_non_3_element_grid(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        with self.assertRaises(ValueError):
            rec().launch(grid=(4, 2), block=(64, 1, 1))

    def test_launch_rejects_non_3_element_block(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        with self.assertRaises(ValueError):
            rec().launch(grid=(4, 2, 1), block=(64,))

    def test_launch_extra_kwargs_ignored(self):
        rec = CuTeDSLKernelRecorder(real_kernel=self._make_kernel())
        rec().launch(grid=(1, 1, 1), block=(32, 1, 1), smem=1024)
        self.assertEqual(rec.grid, (1, 1, 1))
        self.assertEqual(rec.block, (32, 1, 1))

    def test_real_kernel_not_invoked(self):
        called = []
        class Kernel:
            def __call__(self, *a, **kw):
                called.append(a)
        rec = CuTeDSLKernelRecorder(real_kernel=Kernel())
        rec().launch(grid=(1, 1, 1), block=(1, 1, 1))
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
