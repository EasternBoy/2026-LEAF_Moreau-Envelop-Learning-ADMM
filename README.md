# Run scripts
## Whole table
Solving-time / optimality-gap table for the maximum-entropy cone program:
the IPOPT and sLME-ADMM columns.  The DC3 column is filled by entr_max_table.py.

  julia --project=. script/entropy_max/entr_max_table.jl            # use stored data, run what is missing
  julia --project=. script/entropy_max/entr_max_table.jl --force    # recompute everything

Internal mode (one (n, m) per process, because `n` is a `const` in
examples/entr_max/maxEntropy.jl):

  julia --project=. --threads=auto script/entropy_max/entr_max_table.jl worker n m [--instances-only] [--force]

Data and the rendered table live in data/cone_result (see README.md there).


## Each row of table
One (n, m) row block of the maximum-entropy table (entr_max_table.jl) over
N_SAMPLES = 1000 instances: solving time (g_opt ≤ 0.1%), solving time
(g_opt ≤ 1%), Constr. viol. and Opt. gap (%) for IPOPT and sLME-ADMM.

  julia --project=. script/entropy_max/entr_max_row.jl n m            # use stored data, run what is missing
  julia --project=. script/entropy_max/entr_max_row.jl n m --force    # recompute this (n, m)

The DC3 + correction column is read from data/cone_result/DC3-n=..-m=...npz, which
entr_max_table.py writes; it shows — when that file does not exist.
The rows are printed and written to data/cone_result/entr_max_row-n=..-m=...md.