import pickle
import numpy as np
import os
import json
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from micnn_power_grid import train_icnn, batched_forward, batched_grad_wrt_x, to_serializable

if __name__ == "__main__": 
    data_train   = np.load(os.path.join("data", "power_grid", "training", "eco_mpc-rho=1.0-train.npz"))
    data_test    = np.load(os.path.join("data", "power_grid", "training", "eco_mpc-rho=1.0-test.npz"))
    path_to_save = os.path.join("models", "power_grid", "neco_mpc-rho=1")

    Xtr, ytr, gtr = data_train["input"].T, data_train["enve"], data_train["grad"].T
    Xva, yva, gva = data_test["input"].T, data_test["enve"], data_test["grad"].T

    n, N = Xtr.shape[1], Xtr.shape[0]

    params = train_icnn(
        Xtr, ytr, gtr,
        n_in=n,
        widths=[16, 16],
        lr=1e-3,
        grad_weight=5.,   # emphasize gradient matching if you trust gradients
        l2_reg=0,
        batch_size=16,
        epochs=10000,
        seed=0,
    )

    # Quick evaluation
    y_pred = batched_forward(params,    jnp.asarray(Xva))
    g_pred = batched_grad_wrt_x(params, jnp.asarray(Xva))
    val_mse = jnp.mean((y_pred - jnp.asarray(yva)) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1))
    grad_max = jnp.sqrt(jnp.max(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1)))
    print(f"[TEST] value MSE: {val_mse:.4e} | grad MSE: {grad_mse:.4e} | grad MAX: {grad_max:.4e}")


    params["rho"] = data_train["rho"].item()
    params["v"]   = jax.nn.softplus(params["v"])

    for i in range(1,len(params["W"])):
        params["W"][i] = jax.nn.softplus(params["W"][i])

    with open(path_to_save + ".pkl", "wb") as f:
        pickle.dump(params, f)

    with open(path_to_save + ".json", 'w') as f:
        json.dump(to_serializable(params), f)