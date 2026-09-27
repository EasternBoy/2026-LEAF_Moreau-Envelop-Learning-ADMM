import numpy as np
import os
import jax
import jax.numpy as jnp

print(jax.devices())

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from micnn import make_icnn, report_test, save_model
icnn = make_icnn("relu", keep_best=True)

if __name__ == "__main__": 
    data_train   = np.load(os.path.join("data", "mpc", "training", "mpc-train-rho=10.0.npz"))
    data_test    = np.load(os.path.join("data", "mpc", "training", "mpc-test-rho=10.0.npz"))
    path_to_save = os.path.join("models", "mpc", "test")

    Xtr, ytr, gtr = data_train["input"].T, data_train["enve"], data_train["grad"].T
    Xva, yva, gva = data_test["input"].T,  data_test["enve"],  data_test["grad"].T

    n, N = Xtr.shape[1], Xtr.shape[0]

    params = icnn.train_icnn(
        Xtr, ytr, gtr,
        n_in=n,
        widths=[256, 256],
        lr=1e-3,
        grad_weight=5.,   
        l2_reg=1e-5,
        batch_size=256,
        epochs=10000,
        seed=0
    )

    # Quick evaluation
    report_test(icnn, params, Xva, yva, gva)
    # NOTE: trained with the ReLU projection but exported with softplus, as before.
    save_model(params, data_train["rho"].item(), path_to_save, export_act=jax.nn.softplus)
