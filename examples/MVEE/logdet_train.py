import pickle
import numpy as np
import os
import json
import jax
import jax.numpy as jnp

from logdet_mICNNN import train_icnn, batched_forward, batched_grad_wrt_x, to_serializable

if __name__ == "__main__": 
    data_train   = np.load(os.path.join("data/MVEE_data", "logdet-rho=1.0-train-m=50.npz"))
    data_test    = np.load(os.path.join("data/MVEE_data","logdet-rho=1.0-test-m=50.npz"))
    path_to_save = os.path.join("data/MVEE_data", "logdet-rho=1.0-m=50_max_ICNN")

    Xtr, ytr, gtr, ftr = data_train["input"].T, data_train["enve"], data_train["grad"].T, data_train["org_f"] 
    Xva, yva, gva, fva = data_test["input"].T, data_test["enve"], data_test["grad"].T, data_test["org_f"] 

    n, N = Xtr.shape[1], Xtr.shape[0]

    params = train_icnn(
        Xtr, ytr, gtr, ftr,
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
    y_pred = batched_forward(params,    jnp.asarray(Xva))
    g_pred = batched_grad_wrt_x(params, jnp.asarray(Xva))
    # f_val  = jnp.asarray(fva)
 
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