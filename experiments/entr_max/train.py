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
    data_train   = np.load(os.path.join("data", "entr_max", "training", "maxEntropy-manual-rho=1.0-train.npz"))
    data_test    = np.load(os.path.join("data", "entr_max", "training", "maxEntropy-n=1000m=100rho=1.0-test.npz"))
    path_to_save = os.path.join("models", "entr_max", "test")

    Xtr, ytr, gtr = data_train["input"].T, data_train["enve"], data_train["grad"].T
    Xva, yva, gva = data_test["input"].T,  data_test["enve"],  data_test["grad"].T

    n, N = Xtr.shape[1], Xtr.shape[0]

    print(f"Number of data: {N}")

    params = icnn.train_icnn(
        Xtr, ytr, gtr,
        n_in=n,
        widths=[32, 32],
        lr=1e-3,
        grad_weight=5.,   
        l2_reg=0,
        batch_size=256,
        epochs=5000,
        seed=0,
    )

    # Quick evaluation
    report_test(icnn, params, Xva, yva, gva)
    save_model(params, data_train["rho"].item(), path_to_save, export_act=icnn.act_p)
