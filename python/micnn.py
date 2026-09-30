# Input-Convex Neural Network with value + gradient supervision in JAX.
# Convexity is enforced by nonnegative weights (via a projection act_p) on the state
# connections and the last-layer z-weights, and by using a convex, nondecreasing activation.
#
# The problems differ in three training choices, selected with make_icnn / train_icnn:
#   weight_act  "relu" (entr_max, mpc) or "softplus" (power_grid, mvee): the projection act_p
#   keep_best   return the parameters with the best validation objective (entr_max, mpc)
#               or those of the last epoch (power_grid, mvee)
#   f           lower-bound targets: adds penalty_weight * mean(relu(f_pred - f)) and an
#               explicit L2 term l2_reg * ||params||² to the loss (mvee)

import json
import pickle
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

# -----------------------------
# Utils & parameter initialization
# -----------------------------
def glorot_uniform(key, shape):
    fan_in, fan_out = shape[1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def init_icnn_params(
    key: jax.Array,
    n_in: int,
    widths: List[int],
) -> Dict[str, Any]:
    """
    ICNN parameters:
      For L hidden layers with widths = [w1, w2, ..., wL]:
        z0 = act(U0 x + b0)
        zk = act(act_p(Wk) zk-1 + Uk x + bk ),   k=1..L-1
      Output:
        f(x) = act_p(v)^T zL + a^T x + c

      Notes:
        - act is convex & nondecreasing; we use softplus for smooth gradients.
        - act_p(Wk) and act_p(v) ensure elementwise nonnegativity.
        - a^T x + c is linear (convex).
    """
    num_layers = len(widths)
    keys = jax.random.split(key, 3 * num_layers + 3)

    U, W, b = [], [], []
    ki = 0
    for i, w in enumerate(widths):
        # Input skip for every layer
        U.append(glorot_uniform(keys[ki], (w, n_in))); ki += 1

        if i == 0:
            # First layer has no state weight (depends only on x)
            W.append(jax.numpy.zeros((w, n_in)))
        else:
            # Unconstrained state weight; act_p applied in forward pass
            W.append(0.05 * jax.random.normal(keys[ki], (w, widths[i - 1]))); ki += 1

        b.append(jnp.zeros((w,)))

    # Last-layer z-weights (constrained nonnegative via act_p), linear term, and bias
    v = 0.05 * jax.random.normal(keys[-3], (widths[-1],))
    a = 0.01 * jax.random.normal(keys[-2], (n_in,))
    c = jnp.array(0.0)

    return {"U": U, "W": W, "b": b, "v": v, "a": a, "c": c}


WEIGHT_ACTS = {"relu": jax.nn.relu, "softplus": jax.nn.softplus}


def make_icnn(weight_act: str = "relu", keep_best: bool = True) -> SimpleNamespace:
    """The jitted ICNN functions for one choice of weight projection (see the file header)."""
    act_p = WEIGHT_ACTS[weight_act]

    # -----------------------------
    # Forward pass
    # -----------------------------
    @jax.jit
    def icnn_forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        """
        f(x): R^n -> R (scalar)
        x shape: (n,)
        Returns: scalar (0-d array)
        """
        U, W, b = params["U"], params["W"], params["b"]
        act = jax.nn.softplus  # convex & nondecreasing

        z = act(jnp.dot(U[0], x) + b[0])  # first layer (no state W)
        for i in range(1, len(U)):
            # state contribution is constrained nonnegative via act_p(Wi)
            z = act(jnp.dot(act_p(W[i]), z) + jnp.dot(U[i], x) + b[i])

        v_nonneg = act_p(params["v"])  # nonnegative combination of convex features
        return jnp.dot(v_nonneg, z) + jnp.dot(params["a"], x) + params["c"]

    batched_forward = jax.vmap(icnn_forward, in_axes=(None, 0))

    # -----------------------------
    # Gradient wrt input
    # -----------------------------
    @jax.jit
    def grad_wrt_x(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda xx: icnn_forward(params, xx))(x)

    batched_grad_wrt_x = jax.jit(jax.vmap(grad_wrt_x, in_axes=(None, 0)))

    # -----------------------------
    # Loss: value + gradient supervision
    # -----------------------------
    @jax.jit
    def loss_fn(params, xb, yb, gb, grad_weight: float = 1.0):
        """Returns (total_loss, (value_mse, grad_mse))"""
        y_pred = batched_forward(params, xb)             # (B,)
        g_pred = batched_grad_wrt_x(params, xb)          # (B, n)
        value_mse = jnp.mean((y_pred - yb) ** 2)
        grad_mse = jnp.mean(jnp.sum((g_pred - gb) ** 2, axis=1))
        total = value_mse + grad_weight * grad_mse
        return total, (value_mse, grad_mse)

    @jax.jit
    def loss_fn_bounded(params, xb, yb, gb, fb, grad_weight: float = 1.0,
                        penalty_weight: float = 1.0, l2_reg: float = 0.0):
        """Returns (total_loss, (value_mse, grad_mse, lower_bound_penalty))"""
        y_pred = batched_forward(params, xb)
        g_pred = batched_grad_wrt_x(params, xb)
        value_mse = jnp.mean((y_pred - yb) ** 2)
        grad_mse = jnp.mean(jnp.sum((g_pred - gb) ** 2, axis=1))
        lower_bound_penalty = jnp.mean(jax.nn.relu(y_pred - fb))
        l2_penalty = l2_reg * jnp.sum(jnp.array([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)]))
        total = value_mse + grad_weight * grad_mse + penalty_weight * lower_bound_penalty + l2_penalty
        return total, (value_mse, grad_mse, lower_bound_penalty)

    def make_train_step(optimizer, bounded: bool):
        if bounded:
            @jax.jit
            def train_step(params, opt_state, xb, yb, gb, fb, grad_weight, penalty_weight, l2_reg):
                (loss, aux), grads = jax.value_and_grad(loss_fn_bounded, has_aux=True)(
                    params, xb, yb, gb, fb, grad_weight, penalty_weight, l2_reg
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return optax.apply_updates(params, updates), opt_state, loss, aux
        else:
            @jax.jit
            def train_step(params, opt_state, xb, yb, gb, grad_weight):
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, xb, yb, gb, grad_weight
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return optax.apply_updates(params, updates), opt_state, loss, aux
        return train_step

    def train_icnn(
        X: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
        n_in: int,
        widths: List[int] = [64, 64, 64],
        lr: float = 1e-3,
        grad_weight: float = 1.0,
        l2_reg: float = 0,
        batch_size: int = 128,
        epochs: int = 200,
        seed: int = 0,
        f: Optional[np.ndarray] = None,
        penalty_weight: float = 1.0,
    ) -> Dict[str, Any]:
        bounded = f is not None
        key = jax.random.PRNGKey(seed)
        arrays = [jnp.asarray(a, dtype=jnp.float32) for a in ((X, y, g, f) if bounded else (X, y, g))]
        Xj, yj, gj = arrays[:3]

        params = init_icnn_params(key, n_in=n_in, widths=widths)
        best_params = params
        best_val = jnp.inf

        optimizer = optax.adamw(learning_rate=lr, weight_decay=l2_reg)
        opt_state = optimizer.init(params)
        train_step = make_train_step(optimizer, bounded)
        extra = (grad_weight, penalty_weight, l2_reg) if bounded else (grad_weight,)

        for ep in range(1, epochs + 1):
            it_key, key = jax.random.split(key)
            for batch in batch_iterator(arrays, batch_size, it_key):
                params, opt_state, loss, aux = train_step(params, opt_state, *batch, *extra)

            if ep % max(1, epochs // 10) == 0 or ep == 1:
                with jax.disable_jit():
                    yp = batched_forward(params, Xj[:1024])
                    gp = batched_grad_wrt_x(params, Xj[:1024])
                    vm = jnp.mean((yp - yj[:1024]) ** 2)
                    gm = jnp.mean(jnp.sum((gp - gj[:1024]) ** 2, axis=1))
                    val_obj = vm + grad_weight * gm
                    msg = f"Epoch {ep:4d} | val MSE: {vm:.4e} | grad MSE: {gm:.4e}"
                    if bounded:
                        pm = jnp.mean(jax.nn.relu(yp - arrays[3][:1024]))
                        val_obj = val_obj + penalty_weight * pm
                        msg += f" | penalty MSE: {pm:.4e}"
                    if keep_best and val_obj < best_val:
                        best_val = val_obj
                        best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
                    print(msg)

        return best_params if keep_best else params

    def train_icnn_selfsup(
        X: np.ndarray,
        n_in: int,
        objective,
        rho: float,
        widths: List[int] = [64, 64, 64],
        lr: float = 1e-3,
        l2_reg: float = 0,
        batch_size: int = 128,
        epochs: int = 200,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
        label_weight: float = 0.0,
    ) -> Dict[str, Any]:
        """Training without prox solves or gradient labels.  With p = x - ∇f_θ(x)/rho, the prox
        given by ∇ME = rho (x - prox(x)), the loss is
            mean[ objective(p) + rho/2 |p - x|² ]          (minimized at p = prox(x))
          + label_weight · mean[ (f_θ(x) - y)² ]           (given value labels y = ME(x))
        keep_best uses this loss."""

        def losses(params, xb, yb):
            p = xb - batched_grad_wrt_x(params, xb) / rho
            prox_obj = objective(p) + rho / 2 * jnp.sum((p - xb) ** 2, axis=1)
            label_gap = 0.0 if yb is None else jnp.mean((batched_forward(params, xb) - yb) ** 2)
            return jnp.mean(prox_obj), label_gap

        def loss(params, xb, yb):
            po, lg = losses(params, xb, yb)
            return po + label_weight * lg

        key = jax.random.PRNGKey(seed)
        Xj = jnp.asarray(X, dtype=jnp.float32)
        yj = None if y is None else jnp.asarray(y, dtype=jnp.float32)
        arrays = [Xj] if y is None else [Xj, yj]
        params = init_icnn_params(key, n_in=n_in, widths=widths)
        best_params, best_val = params, jnp.inf
        optimizer = optax.adamw(learning_rate=lr, weight_decay=l2_reg)
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state, xb, yb=None):
            grads = jax.grad(loss)(params, xb, yb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state

        for ep in range(1, epochs + 1):
            it_key, key = jax.random.split(key)
            for batch in batch_iterator(arrays, batch_size, it_key):
                params, opt_state = train_step(params, opt_state, *batch)

            if ep % max(1, epochs // 10) == 0 or ep == 1:
                po, lg = losses(params, Xj, yj)
                val_obj = po + label_weight * lg
                if keep_best and val_obj < best_val:
                    best_val = val_obj
                    best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
                print(f"Epoch {ep:4d} | prox objective: {po:.6e} | label gap: {lg:.4e}")

        return best_params if keep_best else params

    return SimpleNamespace(act_p=act_p, icnn_forward=icnn_forward, batched_forward=batched_forward,
                           grad_wrt_x=grad_wrt_x, batched_grad_wrt_x=batched_grad_wrt_x,
                           loss_fn=loss_fn, loss_fn_bounded=loss_fn_bounded, train_icnn=train_icnn,
                           train_icnn_selfsup=train_icnn_selfsup)


# -----------------------------
# Objectives for training without prox solves (train_icnn_selfsup): f summed over the input, per sample
# -----------------------------
def entr_max_objective(S0: float, eps: float = 1e-9):
    """x log(x/S0), the entr_max prox function (problems/entr_max/admm.jl), for x >= eps;
    a steep quadratic below eps, where the prox never lies."""
    def f(p):
        pc = jnp.maximum(p, eps)
        return jnp.sum(pc * jnp.log(pc / S0) + 1e3 * jax.nn.relu(eps - p) ** 2, axis=1)
    return f


OBJECTIVES = {"entr_max": entr_max_objective}


# -----------------------------
# Batch setups
# -----------------------------
def batch_iterator(arrays, batch_size: int, shuffle_key: jax.Array):
    """
    Yields shuffled mini-batches of every array in `arrays` using pure JAX shuffling.
    """
    N = arrays[0].shape[0]
    perm = jax.random.permutation(shuffle_key, N)  # shape (N,)
    for s in range(0, N, batch_size):
        j = perm[s : s + batch_size]
        yield tuple(a[j] for a in arrays)


# -----------------------------
# Evaluation and export (used by train.py)
# -----------------------------
def report_test(icnn: SimpleNamespace, params, Xva, yva, gva) -> None:
    y_pred = icnn.batched_forward(params,    jnp.asarray(Xva))
    g_pred = icnn.batched_grad_wrt_x(params, jnp.asarray(Xva))
    val_mse = jnp.mean((y_pred - jnp.asarray(yva)) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1))
    grad_max = jnp.sqrt(jnp.max(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1)))
    print(f"[TEST] value MSE: {val_mse:.4e} | grad MSE: {grad_mse:.4e} | grad MAX: {grad_max:.4e}")


def save_model(params, rho: float, path_to_save: str, export_act) -> None:
    """Writes <path>.pkl and <path>.json (read by load_model in src/icnn.jl) with the
    projection export_act applied to v and the state weights W[1:], as the Julia ICNN
    uses the weights as they are.  A path ending in .npz writes that one file instead:
    U1.., W1.., b1.. (one per layer), v, a, c and rho, in float64."""
    params["rho"] = rho
    params["v"]   = export_act(params["v"])
    for i in range(1, len(params["W"])):
        params["W"][i] = export_act(params["W"][i])
    if path_to_save.endswith(".npz"):
        arrays = {"v": params["v"], "a": params["a"], "c": params["c"], "rho": rho}
        for key in ("U", "W", "b"):
            arrays.update({f"{key}{i + 1}": x for i, x in enumerate(params[key])})
        np.savez(path_to_save, **{k: np.asarray(x, dtype=np.float64) for k, x in arrays.items()})
        return
    with open(path_to_save + ".pkl", "wb") as f:
        pickle.dump(params, f)
    with open(path_to_save + ".json", 'w') as f:
        json.dump(to_serializable(params), f)


def to_serializable(obj):
    """Recursively convert JAX arrays (or numpy) to lists for JSON."""
    if isinstance(obj, (jax.Array, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    else:
        return obj  # numbers, strings, etc. stay the same
