# Maximum-entropy cone program: n = 100, m = 10

Solving time in ms, Constr. viol. and optimality gap in %, as mean (max) over
1000 instances; **bold** is the lowest mean time in the row; `[k it.]` is
sLME-ADMM's mean number of iterations; — means the data has not been produced yet.
See entr_max_table.md / README.md for the definitions.

|  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|
| solving time (g_opt ≤ 0.1%) | 100 | 10 | 0.92 (1.37) | **0.78 (1.04)** [19.7 it.] | unable to achieve |
| solving time (g_opt ≤ 1%) | 100 | 10 | **0.71 (0.97)** | 0.78 (1.04) [19.7 it.] | unable to achieve |
| Constr. viol. (g_opt ≤ 0.1%) | 100 | 10 | 2.2e-16 (8.9e-16) | 5.3e-06 (9.9e-06) | 4.8e-05 (1.0e-04) |
| Constr. viol. (g_opt ≤ 1%) | 100 | 10 | 2.2e-16 (8.9e-16) | 5.3e-06 (9.9e-06) | 4.8e-05 (1.0e-04) |
| Opt. gap (%) | 100 | 10 | 0 | 0.000475 (0.00539) | 0.837 (16.4) |
