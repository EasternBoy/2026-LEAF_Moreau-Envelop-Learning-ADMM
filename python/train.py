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
    ap.add_argument("--init", choices=("default", "fan_in"), help="override the ICNN initialisation (see init_icnn_params)")
    ap.add_argument("--output", help="override the output path (without extension)")
    a = ap.parse_args()

    with open(os.path.join(CONFIG_DIR, f"{a.problem}.json")) as fh:
        cfg = json.load(fh)
    train_args = dict(cfg["train"])
    if a.epochs is not None:
        train_args["epochs"] = a.epochs
    if a.init is not None:
        train_args["init"] = a.init
    output = a.output or os.path.join(REPO, cfg["output"])

    print(jax.devices())
    data_train = np.load(os.path.join(REPO, cfg["train_data"]))
    data_test = np.load(os.path.join(REPO, cfg["test_data"]))
    Xtr, ytr, gtr = data_train["input"].T, data_train["enve"], data_train["grad"].T
    Xva, yva, gva = data_test["input"].T, data_test["enve"], data_test["grad"].T
    n, N = Xtr.shape[1], Xtr.shape[0]
    labels_test = (yva, gva)   # original units, for the final reports
    # scale_by_rho: train on ME/ρ and ∇ME/ρ (the envelope of f/ρ at ρ = 1, gradient q - prox(q));
    # the saved model has its output layer multiplied by ρ, so it still computes ME.
    out_scale = float(data_train["rho"]) if cfg.get("scale_by_rho", False) else 1.0
    if out_scale != 1.0:
        ytr, gtr, yva, gva = ytr / out_scale, gtr / out_scale, yva / out_scale, gva / out_scale
        print(f"training on ME/rho and grad ME/rho (rho = {out_scale:g}); logged MSEs are in these units (x rho^2 = original)")
    print(f"Number of data: {N}")

    acts = cfg.get("acts")   # per-layer hidden activations (default: softplus everywhere)
    icnn = make_icnn(cfg["weight_act"], keep_best=cfg["keep_best"], acts=acts)
    if "self_supervised" in cfg:   # no gradient labels; the ME labels only with label_weight > 0
        ss = dict(cfg["self_supervised"])
        objective = OBJECTIVES[ss.pop("objective")](**ss)
        use_labels = train_args.get("label_weight", 0) > 0   # the ME labels (not the gradient labels)
        params = icnn.train_icnn_selfsup(Xtr, n_in=n, objective=objective, rho=data_train["rho"].item(),
                                         y=ytr if use_labels else None, **train_args)
    else:
        if cfg["lower_bound"]:
            train_args["f"] = data_train["org_f"]
        if cfg.get("validate_on_test", False):   # select the best epoch on the test data
            train_args["val_data"] = (Xva, yva, gva)
        if train_args.get("save_at"):   # <output>-<epoch>ep.npz at each save_at epoch and at the end
            base = output[:-len(".npz")] if output.endswith(".npz") else output
            output = f"{base}-{train_args['epochs']}ep.npz"

            def on_save(ep, snapshot, best_ep):
                path = f"{base}-{ep}ep.npz"
                print(f"epoch {ep}: model from epoch {best_ep}")
                report_test(icnn, snapshot, Xva, *labels_test, scale=out_scale)
                save_model(snapshot, data_train["rho"].item(), path, export_act=WEIGHT_ACTS[cfg["export_act"]],
                           out_scale=out_scale, acts=acts)
                print(f"saved {path}")
            train_args["on_save"] = on_save
        params = icnn.train_icnn(Xtr, ytr, gtr, n_in=n, **train_args)

    report_test(icnn, params, Xva, *labels_test, scale=out_scale)
    save_model(params, data_train["rho"].item(), output, export_act=WEIGHT_ACTS[cfg["export_act"]],
               out_scale=out_scale, acts=acts)
    print(f"saved {output}" if output.endswith(".npz") else f"saved {output}.json and {output}.pkl")


if __name__ == "__main__":
    main()
