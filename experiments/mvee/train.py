import numpy as np
import os
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from micnn import make_icnn, report_test, save_model
icnn = make_icnn("softplus", keep_best=False)

if __name__ == "__main__": 
    data_train   = np.load(os.path.join("data", "mvee", "training", "logdet-rho=1.0-train-m=50.npz"))
    data_test    = np.load(os.path.join("data", "mvee", "training", "logdet-rho=1.0-test-m=50.npz"))
    path_to_save = os.path.join("models", "mvee", "logdet-rho=1.0-m=50_max_ICNN")

    Xtr, ytr, gtr, ftr = data_train["input"].T, data_train["enve"], data_train["grad"].T, data_train["org_f"] 
    Xva, yva, gva, fva = data_test["input"].T, data_test["enve"], data_test["grad"].T, data_test["org_f"] 

    n, N = Xtr.shape[1], Xtr.shape[0]

    params = icnn.train_icnn(
        Xtr, ytr, gtr, f=ftr,
        n_in=n,
        widths=[16, 16],
        lr=5e-3,
        grad_weight=5,   # emphasize gradient matching if you trust gradients
        penalty_weight=1, 
        l2_reg=0,
        batch_size=16,
        epochs=5000,
        seed=0,
    )

    # Quick evaluation
    report_test(icnn, params, Xva, yva, gva)
    save_model(params, data_train["rho"].item(), path_to_save, export_act=jax.nn.softplus)
