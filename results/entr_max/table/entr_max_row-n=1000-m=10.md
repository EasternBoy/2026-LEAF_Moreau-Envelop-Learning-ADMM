# Maximum-entropy cone program: n = 1000, m = 10

Solving time in ms, Constr. viol. and optimality gap in %, as mean (max) over
1000 instances; **bold** is the lowest mean time in the row; `[k it.]` is
sLME-ADMM's mean number of iterations; — means the data has not been produced yet.
See entr_max_table.md / README.md for the definitions.

|  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|
| solving time (g_opt ≤ 0.1%) | 1000 | 10 | 5.61 (5.98) | **1.41 (1.82)** [10.0 it.] | unable to achieve |
| solving time (g_opt ≤ 1%) | 1000 | 10 | 5.31 (6.25) | **1.41 (1.82)** [10.0 it.] | unable to achieve |
| Constr. viol. (g_opt ≤ 0.1%) | 1000 | 10 | 6.3e-16 (2.7e-15) | 2.3e-06 (3.2e-06) | 1.5e-04 (3.5e-04) |
| Constr. viol. (g_opt ≤ 1%) | 1000 | 10 | 2.3e-09 (2.3e-06) | 2.3e-06 (3.2e-06) | 1.5e-04 (3.5e-04) |
| Opt. gap (%) | 1000 | 10 | 0 | 0.000911 (0.00223) | 2.28 (16.9), feasible 19% |
