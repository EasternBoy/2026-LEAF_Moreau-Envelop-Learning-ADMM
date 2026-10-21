# Power-grid benchmark, N = 192 (576 scalar variables), CPU, 1000 instances

All cells are mean (max). Time: solving time in ms; gap: optimality gap in %; viol.: largest violation of the original constraints. **Bold**: lowest mean solving time for that target. DC3 + correction has no gap-based stop: one run, shown under both targets, "Unable" when its maximum gap exceeds the target.

| Solver | g_opt ≤ 1%: time | g_opt ≤ 1%: gap | g_opt ≤ 1%: viol. | g_opt ≤ 0.01%: time | g_opt ≤ 0.01%: gap | g_opt ≤ 0.01%: viol. |
|---|---:|---:|---:|---:|---:|---:|
| ADMM | 271.63 (342.84) | 8.6e-01 (1.0e+00) | 2.3e-15 (6.5e-14) | 471.91 (668.60) | 9.7e-03 (1.0e-02) | 3.9e-15 (5.4e-14) |
| IPOPT | 25.50 (110.14) | 5.8e-03 (6.7e-03) | 3.1e-17 (4.9e-17) | 27.23 (111.41) | 3.2e-03 (3.9e-03) | 2.9e-17 (4.2e-17) |
| MadNLP | 18.83 (123.22) | 5.6e-03 (6.7e-03) | 3.2e-14 (1.8e-13) | 20.28 (124.90) | 2.9e-03 (3.9e-03) | 3.6e-15 (1.7e-14) |
| DC3 + correction | Unable | 7.5e-02 (7.4e+00) | 1.2e-04 (1.2e-04) | Unable | 7.5e-02 (7.4e+00) | 1.2e-04 (1.2e-04) |
| sMEL-ADMM | **1.38 (2.39)** | 1.8e-01 (9.9e-01) | 3.0e-14 (3.2e-14) | **2.76 (4.12)** | 8.4e-03 (9.0e-03) | 3.0e-14 (3.2e-14) |
