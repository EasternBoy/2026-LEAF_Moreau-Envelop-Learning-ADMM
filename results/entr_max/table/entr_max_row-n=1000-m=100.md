# Maximum-entropy cone program: n = 1000, m = 100

Solving time in ms, Constr. viol. and optimality gap in %, as mean (max) over
1000 instances; **bold** is the lowest mean time in the row; `[k it.]` is
sLME-ADMM's mean number of iterations; — means the data has not been produced yet.
See entr_max_table.md / README.md for the definitions.

|  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|
| solving time (g_opt ≤ 0.1%) | 1000 | 100 | 50.27 (66.12) | **11.87 (16.06)** [42.0 it.] | unable to achieve |
| solving time (g_opt ≤ 1%) | 1000 | 100 | 46.26 (62.00) | **11.87 (16.06)** [42.0 it.] | unable to achieve |
| Constr. viol. (g_opt ≤ 0.1%) | 1000 | 100 | 5.5e-09 (5.5e-06) | 2.8e-07 (5.0e-06) | 1.5e-03 (2.5e-03) |
| Constr. viol. (g_opt ≤ 1%) | 1000 | 100 | 2.1e-08 (8.4e-06) | 2.8e-07 (5.0e-06) | 1.5e-03 (2.5e-03) |
| Opt. gap (%) | 1000 | 100 | 0 | 0.00812 (0.0574) | 8.37 (12.5), feasible 0% |
