import unittest
import tempfile

import numpy as np
import torch

from DC3.common.runner import _dc3_eval, _latency, build, load_config
from DC3.qp import data as D
from DC3.qp.experiment import SPEC
from DC3.qp.reference import solve_instance


class QPTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(SPEC, None)
        self.cfg["dc3"]["device"] = "cpu"
        self.problem, self.completion, self.solver, _, self.device, self.dtype = build(SPEC, self.cfg)
        self.params = SPEC.build_split(self.cfg["data"], "train", 3, self.device, self.dtype)

    def test_completion_and_objective(self):
        Z = torch.randn(3, self.completion.n_partial, dtype=self.dtype)
        Y = self.completion.complete(Z, self.params.x)
        self.assertLess(self.problem.eq_resid(self.params, Y).abs().max().item(), 1e-10)
        matrices = D.fixed_data()
        expected = np.array([0.5 * y @ matrices["Q"] @ y + matrices["p"] @ y for y in Y.numpy()])
        np.testing.assert_allclose(self.problem.obj_fn(self.params, Y).numpy(), expected)
        np.testing.assert_allclose(self.problem.ineq_resid(self.params, Y).numpy(),
                                   Y.numpy() @ matrices["G"].T - matrices["h"])

    def test_closed_form_gradient(self):
        Z = torch.randn(3, self.completion.n_partial, dtype=self.dtype, requires_grad=True)
        Y = self.completion.complete(Z, self.params.x)
        expected, = torch.autograd.grad(self.problem.ineq_dist(self.params, Y).square().sum(), Z)
        actual = self.problem.ineq_partial_grad(self.params, Z, self.completion)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-10)

    def test_splits_and_saved_inputs(self):
        dc = self.cfg["data"]
        train = SPEC.build_split(dc, "train", 3, self.device, self.dtype)
        valid = SPEC.build_split(dc, "valid", 3, self.device, self.dtype)
        torch.testing.assert_close(train.x, self.params.x)
        self.assertFalse(torch.equal(train.x, valid.x))
        path = D.REPO_ROOT / dc["test_instances"]
        if not path.exists():
            self.skipTest("Julia QP test instances have not been generated")
        test = SPEC.build_split(dc, "test", 3, self.device, self.dtype)
        with np.load(path) as saved:
            np.testing.assert_array_equal(test.x.numpy(), saved["X"][:3])

    def test_reference_and_solver(self):
        y, _, J, status = solve_instance(self.params.x[0].numpy())
        self.assertEqual(status, "optimal")
        Y = torch.as_tensor(y[None, :], dtype=self.dtype)
        params = self.params.index(slice(0, 1))
        self.assertLess(self.problem.eq_resid(params, Y).abs().max().item(), 1e-7)
        self.assertLess(self.problem.ineq_resid(params, Y).max().item(), 1e-7)
        self.assertAlmostEqual(self.problem.obj_fn(params, Y).item(), J, places=8)
        result = self.solver.solve(self.params)
        self.assertEqual(result["Y"].shape, (3, 100))
        self.assertTrue(torch.isfinite(result["Y"]).all())
        self.assertLess(self.problem.eq_resid(self.params, result["Y"]).abs().max().item(), 1e-9)
        self.assertLessEqual(result["steps"], self.cfg["dc3"]["corr_test_max_steps"])

    def test_benchmark_interfaces(self):
        evaluation = _dc3_eval(self.solver, self.problem, self.params, self.params, None,
                               1e-4, SPEC.index_fn, self.device, n_warmup=1)
        self.assertEqual(evaluation["time_s"].shape, (3,))
        latency = _latency(SPEC, self.solver, self.params, self.device, (1, 3), 1, 1)
        self.assertEqual(latency["batch_3"]["batch_size"], 3)
        self.assertGreater(latency["batch_3"]["median_ms"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = SPEC.export_instances(self.params, directory, self.cfg)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["X"], self.params.x.numpy())
                for key, value in D.fixed_data().items():
                    np.testing.assert_array_equal(saved[key], value)


if __name__ == "__main__":
    unittest.main()
