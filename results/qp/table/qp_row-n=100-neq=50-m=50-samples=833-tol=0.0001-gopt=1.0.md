# QP: n = 100, neq = 50, m = 50

Solving time in ms and optimality gap in %, as mean (maximum) over
833 instances. Constraint violation is the maximum original-coordinate
equality/inequality violation per instance, also shown as mean (maximum).
**Bold** marks the lowest mean time among methods whose every saved solution
meets the 1.0% objective-gap target and feasibility threshold c_v = 0.001.
"unable to achieve" means a target was missed; — means results are missing.
Timings retain the boundaries described in README.md; this script does not retrain DC3.

|  | n | neq | m | OSQP mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|---|
| Solving time (g_opt ≤ 1.0%) | 100 | 50 | 50 | **3.539 (4.687)** | unable to achieve | unable to achieve |
| Constr. viol. | 100 | 50 | 50 | 1.6e-08 (1.5e-07) | 7.5e-04 (1.1e-03) | 1.1e-13 (1.4e-13) |
| Opt. gap (%) | 100 | 50 | 50 | 4.01e-14 (1.85e-13) | 0.00342 (0.0161) | 5.62 (6.88) |
