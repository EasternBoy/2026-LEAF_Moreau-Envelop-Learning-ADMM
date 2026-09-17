# Input-Convex Neural Network with value + gradient supervision in JAX.
# Convexity is enforced by nonnegative weights (via softplus) on the state connections
# and the last-layer z-weights, and by using a convex, nondecreasing activation.

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Tuple
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
        zk = act(softplus(Wk) zk-1 + Uk x + bk ),   k=1..L-1
      Output:
        f(x) = softplus(v)^T zL + a^T x + c

      Notes:
        - act is convex & nondecreasing; we use softplus for smooth gradients.
        - softplus(Wk) and softplus(v) ensure elementwise nonnegativity.
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
            # Unconstrained state weight; softplus/relu/exp applied in forward pass
            W.append(0.05 * jax.random.normal(keys[ki], (w, widths[i - 1]))); ki += 1

        b.append(jnp.zeros((w,)))

    # Last-layer z-weights (constrained nonnegative via softplus), linear term, and bias
    v = 0.05 * jax.random.normal(keys[-3], (widths[-1],))
    a = 0.01 * jax.random.normal(keys[-2], (n_in,))
    c = jnp.array(0.0)

    return {"U": U, "W": W, "b": b, "v": v, "a": a, "c": c}


def act_p(x):
    """Nonnegative projection for weights. Can be softplus or ReLU."""
    return jax.nn.relu(x)  # ReLU is nonnegative but has zero-gradient regions; use with caution.

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
    act   = jax.nn.softplus  # convex & nondecreasing
    # act_p = jax.nn.relu      # nonnegative projection for weights

    z = act(jnp.dot(U[0], x) + b[0])  # first layer (no state W)
    for i in range(1, len(U)):
        Wi = W[i]
        # state contribution is constrained nonnegative via softplus(Wi)
        z = act(jnp.dot(act_p(Wi), z) + jnp.dot(U[i], x) + b[i])

    v_nonneg = act_p(params["v"])  # nonnegative combination of convex features
    f = jnp.dot(v_nonneg, z) + jnp.dot(params["a"], x) + params["c"]
    return f  # scalar (shape: ())


# Vectorized forward for batches
batched_forward = jax.vmap(icnn_forward, in_axes=(None, 0))


# -----------------------------
# Gradient wrt input
# -----------------------------
@jax.jit
def grad_wrt_x(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    # x shape: (n,)
    return jax.grad(lambda xx: icnn_forward(params, xx))(x)


batched_grad_wrt_x = jax.jit(jax.vmap(grad_wrt_x, in_axes=(None, 0)))


# -----------------------------
# Loss: value + gradient supervision
# -----------------------------
@jax.jit
def loss_fn(
    params: Dict[str, Any],
    xb: jnp.ndarray,  # (B, n)
    yb: jnp.ndarray,  # (B,)
    gb: jnp.ndarray,  # (B, n)
    grad_weight: float = 1.0
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Returns (total_loss, (value_mse, grad_mse))
    """
    y_pred = batched_forward(params, xb)             # (B,)
    g_pred = batched_grad_wrt_x(params, xb)          # (B, n)

    value_mse = jnp.mean((y_pred - yb) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - gb) ** 2, axis=1))

    total = value_mse + grad_weight * grad_mse
    return total, (value_mse, grad_mse)

# -----------------------------
# Batch setups
# -----------------------------
def batch_iterator(X: jnp.ndarray, y: jnp.ndarray, g: jnp.ndarray, batch_size: int, shuffle_key: jax.Array):
    """
    Yields shuffled mini-batches using pure JAX shuffling (no Python int conversion).
    """
    N = X.shape[0]
    perm = jax.random.permutation(shuffle_key, N)  # shape (N,)
    for s in range(0, N, batch_size):
        j = perm[s : s + batch_size]
        yield X[j], y[j], g[j]

def make_train_step(optimizer):
    @jax.jit
    def train_step(
        params: Dict[str, Any],
        opt_state: optax.OptState,
        xb: jnp.ndarray,
        yb: jnp.ndarray,
        gb: jnp.ndarray,
        grad_weight: float,
    ):
        (loss, (vmse, gmse)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, xb, yb, gb, grad_weight
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, vmse, gmse
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
) -> Dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    Xj = jnp.asarray(X, dtype=jnp.float32)
    yj = jnp.asarray(y, dtype=jnp.float32)
    gj = jnp.asarray(g, dtype=jnp.float32)

    params = init_icnn_params(key, n_in=n_in, widths=widths)
    best_params = params
    best_val = jnp.inf

    # --- NEW: Adam optimizer ---
    optimizer = optax.adamw(learning_rate=lr, weight_decay=l2_reg)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    for ep in range(1, epochs + 1):
        it_key, key = jax.random.split(key)
        for xb, yb, gb in batch_iterator(Xj, yj, gj, batch_size, it_key):
            params, opt_state, loss, vmse, gmse = train_step(
                params, opt_state, xb, yb, gb, grad_weight
            )

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            with jax.disable_jit():
                yp = batched_forward(params, Xj[:1024])
                gp = batched_grad_wrt_x(params, Xj[:1024])
                vm = jnp.mean((yp - yj[:1024]) ** 2)
                gm = jnp.mean(jnp.sum((gp - gj[:1024]) ** 2, axis=1))
                val_obj = vm + grad_weight * gm
                if val_obj < best_val:
                    best_val = val_obj
                    best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
                print(f"Epoch {ep:4d} | val MSE: {vm:.4e} | grad MSE: {gm:.4e}")

    return best_params

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
