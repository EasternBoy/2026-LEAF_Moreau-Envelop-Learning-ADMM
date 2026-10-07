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
| Solving time (g_opt ≤ 1.0%) | 100 | 50 | 50 | **3.457 (5.022)** | 5.798 (12.731) | unable to achieve |
| Constr. viol. | 100 | 50 | 50 | 1.7e-08 (1.8e-07) | 4.6e-04 (5.3e-04) | 1.1e-13 (1.4e-13) |
| Opt. gap (%) | 100 | 50 | 50 | 3.99e-14 (1.85e-13) | 0.0022 (0.0176) | 5.62 (6.88) |
