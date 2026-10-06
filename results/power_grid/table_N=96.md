# Power-grid benchmark, N = 96 (288 scalar variables), CPU, 1000 instances

All cells are mean (max). Time: solving time in ms; gap: optimality gap in %; viol.: largest violation of the original constraints. **Bold**: lowest mean solving time for that target. DC3 + correction has no gap-based stop: one run, shown under both targets, "Unable" when its maximum gap exceeds the target.

| Solver | g_opt ≤ 1%: time | g_opt ≤ 1%: gap | g_opt ≤ 1%: viol. | g_opt ≤ 0.01%: time | g_opt ≤ 0.01%: gap | g_opt ≤ 0.01%: viol. |
|---|---:|---:|---:|---:|---:|---:|
| ADMM | 125.11 (190.76) | 8.9e-01 (1.0e+00) | 4.7e-11 (1.7e-09) | 208.01 (1633.78) | 9.3e-03 (1.0e-02) | 1.6e-11 (6.5e-10) |
| IPOPT | 11.63 (37.57) | 5.9e-03 (7.2e-03) | 2.6e-17 (4.2e-17) | 12.81 (75.75) | 2.7e-03 (3.9e-03) | 2.5e-17 (3.8e-17) |
| MadNLP | 8.93 (22.09) | 5.8e-03 (7.2e-03) | 3.3e-14 (1.7e-13) | 9.92 (58.80) | 2.5e-03 (3.9e-03) | 4.7e-15 (1.4e-14) |
| DC3 + correction | 7.37 (13.70) | 1.5e-01 (6.2e-01) | 9.7e-05 (1.0e-04) | Unable | 1.5e-01 (6.2e-01) | 9.7e-05 (1.0e-04) |
| MEL-ADMM | 18.31 (27.05) | 9.0e-01 (1.0e+00) | 3.5e-11 (1.3e-09) | 32.64 (224.37) | 9.3e-03 (1.0e-02) | 1.3e-11 (6.7e-10) |
| sMEL-ADMM | **1.44 (2.34)** | 1.2e-01 (1.0e+00) | 1.2e-14 (1.4e-14) | **3.39 (22.76)** | 5.1e-03 (6.5e-03) | 2.2e-05 (2.7e-05) |
