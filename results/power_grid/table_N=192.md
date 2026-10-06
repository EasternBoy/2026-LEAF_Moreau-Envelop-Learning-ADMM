# Power-grid benchmark, N = 192 (576 scalar variables), CPU, 1000 instances

All cells are mean (max). Time: solving time in ms; gap: optimality gap in %; viol.: largest violation of the original constraints. **Bold**: lowest mean solving time for that target. DC3 + correction has no gap-based stop: one run, shown under both targets, "Unable" when its maximum gap exceeds the target.

| Solver | g_opt ≤ 1%: time | g_opt ≤ 1%: gap | g_opt ≤ 1%: viol. | g_opt ≤ 0.01%: time | g_opt ≤ 0.01%: gap | g_opt ≤ 0.01%: viol. |
|---|---:|---:|---:|---:|---:|---:|
| ADMM | 269.56 (1605.83) | 8.6e-01 (1.0e+00) | 2.0e-11 (2.0e-09) | 444.03 (619.14) | 9.7e-03 (1.0e-02) | 1.8e-11 (8.4e-10) |
| IPOPT | 26.39 (214.07) | 5.8e-03 (6.7e-03) | 3.1e-17 (4.9e-17) | 26.67 (111.76) | 3.2e-03 (3.9e-03) | 2.9e-17 (4.2e-17) |
| MadNLP | 20.71 (188.21) | 5.6e-03 (6.7e-03) | 2.9e-14 (1.5e-13) | 21.05 (131.74) | 2.9e-03 (3.9e-03) | 3.6e-15 (1.2e-14) |
| DC3 + correction | Unable | 7.5e-02 (7.4e+00) | 1.2e-04 (1.2e-04) | Unable | 7.5e-02 (7.4e+00) | 1.2e-04 (1.2e-04) |
| MEL-ADMM | 37.30 (218.78) | 8.7e-01 (1.0e+00) | 1.4e-11 (1.0e-09) | 65.40 (88.15) | 9.6e-03 (1.0e-02) | 1.0e-11 (1.0e-09) |
| sMEL-ADMM | **2.13 (15.47)** | 1.8e-01 (9.9e-01) | 2.9e-14 (3.2e-14) | **4.37 (5.00)** | 8.4e-03 (9.0e-03) | 2.9e-14 (3.1e-14) |
