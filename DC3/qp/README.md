# QP DC3 benchmark

This module follows `DC3/entr_max`: the same training and benchmark entry points,
`AppSpec`, shared network, equality completion, correction, metrics, and runner.
Only the problem, data, reference solver, and configuration are QP-specific.

The formulation matches `problems/qp/problem.jl`:

    min_y  0.5 y'Qy + p'y    subject to Ay=x, Gy<=h

The matrices use the original DC3 NumPy seed 17 and draw order. Only x varies.
Training and validation use separate reproducible NumPy streams; testing reads
the exact saved inputs used by `experiments/qp/table.jl`:

    results/qp/table/instances/instances-n=100-neq=50-m=50-samples=833.npz

The test loader verifies all five fixed matrices before using the file. If the
file is missing, run the Julia QP table first. Change `data.test_instances` and
`data.n_test` together when using another saved set. The benchmark also exports
these inputs to `DC3/results/qp-default/test_instances.npz`.

Run from the repository root, in the same Python environment used for entropy:

```sh
python3 -m DC3.qp.train --tag default
python3 -m DC3.qp.benchmark --tag default --batch-sizes 1 8 32 100 833
```

The default device is CPU for the Apple M4 Pro setup. To use NVIDIA CUDA, run on a CUDA
machine with a CUDA-enabled PyTorch installation:

```sh
python3 -m DC3.qp.train --tag cuda --set dc3.device=cuda
python3 -m DC3.qp.benchmark --tag cuda --set dc3.device=cuda --batch-sizes 1 8 32 100 833
```

If overriding with `device=auto`, inspect the printed device and saved environment.
Float64 uses CPU rather than MPS on Macs. Dependencies are the same as entropy
(`DC3/requirements.txt`). Configuration overrides use the existing `--set` syntax.

The default config uses two hidden layers of 512 ReLU units with batch
normalization and no dropout, Adam at learning rate 1e-3, 5000 epochs, and training
batch size 64. It retains the QP correction settings: ten training correction
steps, at most ten test correction steps, and correction tolerance 1e-4. It does
not include entropy-specific positivity transforms, initialization targets, or
correction preconditioning. The shared runner and new train/validation/test
inputs and network settings make this a local adaptation, not an exact reproduction of the paper.
Reaching the correction cap does not establish feasibility; check the reports.

Results go to `DC3/results/qp-default/` (or `qp-cuda/`):

- `checkpoint.pt` and `train_history.json`: training outputs.
- `benchmark.json`, `per_instance.csv`, `REPORT.md`, `summary.csv`: benchmark outputs.
- `benchmark.json` contains `dc3.latency.single_instance` for matched batch-1
  latency and quality, plus `dc3.latency.batch_833` for full-batch latency and
  throughput. `median_ms` in `batch_833` measures all 833 instances together;
  it is not directly comparable with the Julia table's mean per-instance seconds.
  Use `dc3.latency.single_instance.mean_ms / 1000` for the corresponding DC3
  per-instance mean; note the different internal solver timing boundaries.

The shared runner warms up and synchronizes CUDA around timing. Batch timing
includes prediction, completion, and correction, with inputs already on-device.
Its headline quality is measured from batch-1 calls; correction stops across
the whole batch, so quality of a full-batch output need not equal batch-1 quality.

Run the QP integration checks with:

```sh
python3 -m unittest DC3.tests.test_qp
```
