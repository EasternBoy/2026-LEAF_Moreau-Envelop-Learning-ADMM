"""Train the ICNN model of a problem's Moreau envelope and save it for the Julia code.

    python python/train.py entr_max                  # settings from python/configs/entr_max.json
    python python/train.py mvee --epochs 10 --output /tmp/mvee_quick

Problems: entr_max, mpc, power_grid, mvee.  The config sets the data, the output path
(<output>.json is read by load_model in src/icnn.jl, <output>.pkl keeps the JAX
parameters; an output ending in .npz is written as that single file, which load_model
also reads), the training choices of make_icnn (weight_act, keep_best), the projection
applied to the exported weights (export_act), whether the loss uses the lower bounds
`org_f` of the data (lower_bound), and the arguments of train_icnn (train).  A config with
"self_supervised": {"objective": <name in micnn.OBJECTIVES>, ...its arguments} trains without
prox solves or gradient labels instead (train_icnn_selfsup, whose arguments are then in train);
a "label_weight" > 0 in train adds the ME labels as a supervised value term.  The labels of the
test data are still used by report_test.
Paths in a config are relative to the repository root.
"""

import argparse
import json
import os

import jax
import numpy as np

from micnn import OBJECTIVES, WEIGHT_ACTS, make_icnn, report_test, save_model

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", help="name of a config in python/configs (entr_max, mpc, power_grid, mvee)")
    ap.add_argument("--epochs", type=int, help="override the number of epochs")
    ap.add_argument("--output", help="override the output path (without extension)")
    a = ap.parse_args()

    with open(os.path.join(CONFIG_DIR, f"{a.problem}.json")) as fh:
        cfg = json.load(fh)
    train_args = dict(cfg["train"])
    if a.epochs is not None:
        train_args["epochs"] = a.epochs
    output = a.output or os.path.join(REPO, cfg["output"])

    print(jax.devices())
    data_train = np.load(os.path.join(REPO, cfg["train_data"]))
    data_test = np.load(os.path.join(REPO, cfg["test_data"]))
    Xtr, ytr, gtr = data_train["input"].T, data_train["enve"], data_train["grad"].T
    Xva, yva, gva = data_test["input"].T, data_test["enve"], data_test["grad"].T
    n, N = Xtr.shape[1], Xtr.shape[0]
    print(f"Number of data: {N}")

    icnn = make_icnn(cfg["weight_act"], keep_best=cfg["keep_best"])
    if "self_supervised" in cfg:   # no gradient labels; the ME labels only with label_weight > 0
        ss = dict(cfg["self_supervised"])
        objective = OBJECTIVES[ss.pop("objective")](**ss)
        use_labels = train_args.get("label_weight", 0) > 0   # the ME labels (not the gradient labels)
        params = icnn.train_icnn_selfsup(Xtr, n_in=n, objective=objective, rho=data_train["rho"].item(),
                                         y=ytr if use_labels else None, **train_args)
    else:
        if cfg["lower_bound"]:
            train_args["f"] = data_train["org_f"]
        params = icnn.train_icnn(Xtr, ytr, gtr, n_in=n, **train_args)

    report_test(icnn, params, Xva, yva, gva)
    save_model(params, data_train["rho"].item(), output, export_act=WEIGHT_ACTS[cfg["export_act"]])
    print(f"saved {output}" if output.endswith(".npz") else f"saved {output}.json and {output}.pkl")


if __name__ == "__main__":
    main()
