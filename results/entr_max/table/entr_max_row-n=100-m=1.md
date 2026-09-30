# Maximum-entropy cone program: n = 100, m = 1

Solving time in ms, Constr. viol. and optimality gap in %, as mean (max) over
1000 instances; **bold** is the lowest mean time in the row; `[k it.]` is
sLME-ADMM's mean number of iterations; — means the data has not been produced yet.
See entr_max_table.md / README.md for the definitions.

|  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|
| solving time (g_opt ≤ 0.1%) | 100 | 1 | 0.53 (0.79) | **0.52 (0.95)** [15.1 it.] | unable to achieve |
| solving time (g_opt ≤ 1%) | 100 | 1 | 0.53 (0.81) | 0.52 (0.95) [15.1 it.] | **0.31 (0.60)** |
| Constr. viol. (g_opt ≤ 0.1%) | 100 | 1 | 2.1e-16 (1.1e-15) | 1.9e-06 (1.0e-05) | 1.0e-05 (9.9e-05) |
| Constr. viol. (g_opt ≤ 1%) | 100 | 1 | 2.1e-16 (1.1e-15) | 1.9e-06 (1.0e-05) | 1.0e-05 (9.9e-05) |
| Opt. gap (%) | 100 | 1 | 0 | 5.1e-05 (0.000106) | 0.0399 (0.242) |
