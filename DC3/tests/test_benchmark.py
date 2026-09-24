"""Regression checks for domain validity and matched timing/quality."""
import unittest
import numpy as np
import torch
from DC3.common.metrics import per_instance_metrics, aggregate
from DC3.common.runner import _dc3_eval
from DC3.cone_programming.problem import MaxEntropyProblem, ConeParams
from DC3.power_grid.problem import EcoMPCProblem, GridParams

class BenchmarkTests(unittest.TestCase):
    def test_entropy_domain_is_not_feasibility_tolerance(self):
        p = MaxEntropyProblem(2, 1)
        params = ConeParams(torch.zeros(3, 1, 2, dtype=torch.float64), torch.ones(3, 1, dtype=torch.float64))
        y = torch.tensor([[-1e-5, 1+1e-5], [0, 1], [0.5, 0.5]], dtype=torch.float64)
        m = per_instance_metrics(p, params, y, np.full(3, -np.log(2)))
        np.testing.assert_array_equal(m['constraint_feasible'], [1, 1, 1])
        np.testing.assert_array_equal(m['domain_valid'], [0, 1, 1])
        np.testing.assert_array_equal(m['feasible'], [0, 1, 1])
        self.assertTrue(np.isnan(m['obj'][0]) and np.isnan(m['gap_pct'][0]))
        self.assertEqual(m['obj'][1], 0)  # 0 log 0
        a = aggregate(m)
        self.assertTrue(np.isnan(a['gap_pct_mean']))
        self.assertEqual(a['domain_valid_count'], 2)
        self.assertTrue(np.isfinite(a['gap_pct_feasible_mean']))

    def test_power_domain_strict_positive_and_exact_objective(self):
        p = EcoMPCProblem(N=1)
        params = GridParams(torch.tensor([.5]*3), torch.zeros(3, 1), torch.zeros(3, 1))
        y = torch.tensor([[0.,0.,0.,.5], [0.,0.,-1e-8,.5], [0.,0.,1e-12,.5]], dtype=torch.float64)
        m = per_instance_metrics(p, params, y)
        np.testing.assert_array_equal(m['domain_valid'], [0, 0, 1])
        self.assertGreater(m['obj'][2], m['surrogate_obj'][2]*100)

    def test_quality_comes_from_each_timed_output(self):
        p = MaxEntropyProblem(2, 1)
        params = ConeParams(torch.zeros(3, 1, 2, dtype=torch.float64), torch.ones(3, 1, dtype=torch.float64))
        class Solver:
            calls = 0
            def solve(self, pb):
                self.calls += 1
                assert len(pb) == 1
                y = torch.tensor([[self.calls / 10, 1-self.calls / 10]], dtype=torch.float64)
                return dict(Y=y, Y_raw=y, steps=self.calls, converged=torch.tensor([True]), first_feasible_step=torch.tensor([0]))
        solver = Solver()
        result = _dc3_eval(solver, p, params, params, None, 1e-4, lambda p,i:p.index(i), torch.device('cpu'), n_warmup=1)
        self.assertEqual(solver.calls, 4)
        np.testing.assert_allclose(result['Y'][:, 0], [.2,.3,.4])
        np.testing.assert_array_equal(result['corr_steps'], [2,3,4])
        self.assertEqual(len(result['time_s']), 3)
        self.assertTrue((result['time_s'] >= 0).all())

if __name__ == '__main__':
    unittest.main()
