# Moreau-envelope-learning-for-accelerated-ADMM
Code and related materials for learning the Moreau envelope to accelerate ADMM solving.

## Instructions
Organize the directories in this repo in a systematic manner. Create a directory for each major general part of the code (like ADMM algorithm, NN learning), and a directory for each major example/application.
Do not place all code files in the root directory.
Try your best to avoid a future major reorganization of the code.


## To crate dataset for training and testing a NN model.
At this time, it is temporary.

Set system parameters in src/data_gen/system.jl
Set functions in src/data_gen/cost_Func.jl

Run src/data_gen/data_gen.jl to create dataset and testset
To train NN, run src/ADMM_learning/myICNN_JAX.py
Run src/ADMM_learning/learning_ADMM.py to work with learning in ADMM iteration

Compare the outcome to the optimal solution in src/data_gen/check.jl

Good luck with that :D

## Benchmark for Economic MPC example
This benchmark compares three optimization approaches for **Economic Model Predictive Control (EcoMPC)**:

- **Centralized Solver** (using `MadNLP`)
- **Standard ADMM**
- **Learning-based ADMM** 

All methods solve the same constrained economic optimization problem over a prediction horizon (N = 24), and performance is evaluated in terms of:
- Objective value (`J`)
- Relative optimality gap (%)
- Runtime (via `BenchmarkTools`)
### How to run the benchmark
- The file `EcoMPC_benchmark.jl` is located in the `src/benchmarks` directory.
- From the **Julia REPL**, run the benchmark by including the script `include("src/benchmarks/EcoMPC_benchmark.jl")`





