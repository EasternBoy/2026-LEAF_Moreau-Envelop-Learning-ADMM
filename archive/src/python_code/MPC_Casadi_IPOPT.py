# mpc_l1_ipopt.py
import numpy as np
import casadi as ca

# ---------------- Problem data (from MATLAB) ----------------
A = np.array([[2.0, -1.0],
              [1.0,  0.2]])
B = np.array([[1.0],
              [0.0]])
nx, nu = 2, 1

Q = np.eye(2)        # used inside ||Q x_k||_1
R = 2.0
N = 7

x0 = np.array([3.0, 1.0])

x_min, x_max = -5.0, 5.0
u_min, u_max = -1.0, 1.0

# ---------------- Variable stacking ----------------
# z = [x_1..x_N, u_1..u_N, t_x_1..t_x_N, t_u_1..t_u_N]
nx_blk = nx * N
nu_blk = nu * N
ntx_blk = nx * N
ntu_blk = nu * N
n = nx_blk + nu_blk + ntx_blk + ntu_blk

def ix_x(k):   # 1..N → slice for x_k in z
    s = (k-1)*nx
    return slice(s, s+nx)

def ix_u(k):   # 1..N → slice for u_k
    s = nx_blk + (k-1)*nu
    return slice(s, s+nu)

def ix_tx(k):  # 1..N → slice for t_x,k
    s = nx_blk + nu_blk + (k-1)*nx
    return slice(s, s+nx)

def ix_tu(k):  # 1..N → slice for t_u,k
    s = nx_blk + nu_blk + ntx_blk + (k-1)*nu
    return slice(s, s+nu)

# ---------------- Linear objective via auxiliaries ----------------
# L1 cost: sum ||Q x_k||_1 + R*|u_k|
# Implemented with t_x >= ±Qx and t_u >= ±u, minimize 1^T t_x + R 1^T t_u
g_lin = np.zeros((n, 1))
for k in range(1, N+1):
    g_lin[ix_tx(k)] = 1.0
    g_lin[ix_tu(k)] = R

# ---------------- Variable bounds ----------------
lbx_num = -np.inf * np.ones((n, 1))
ubx_num =  np.inf * np.ones((n, 1))

# x bounds
for k in range(1, N+1):
    lbx_num[ix_x(k)] = x_min
    ubx_num[ix_x(k)] = x_max

# u bounds
for k in range(1, N+1):
    lbx_num[ix_u(k)] = u_min
    ubx_num[ix_u(k)] = u_max

# t_x, t_u >= 0
for k in range(1, N+1):
    lbx_num[ix_tx(k)] = 0.0
    lbx_num[ix_tu(k)] = 0.0

# ---------------- Linear constraints rows: lba <= A_mat z <= uba ----------------
rows = []
lba_list = []
uba_list = []

def push_row(cols, vals, low, up):
    """Add one linear constraint row: sum_j vals[j] * z[cols[j]] in [low, up]"""
    r = ca.SX.zeros(1, n)
    for cj, vj in zip(cols, vals):
        r[0, cj] = vj
    rows.append(ca.DM(r))
    lba_list.append(low)
    uba_list.append(up)

def add_vec_eq(col_blocks, vec_rhs):
    """Add vector equalities: (sum_j M_j * z[col_block_j]) == rhs (componentwise)."""
    for i in range(vec_rhs.shape[0]):
        cols, vals = [], []
        for (start, M) in col_blocks:
            for j in range(M.shape[1]):
                v = M[i, j]
                if v != 0.0:
                    cols.append(start + j)
                    vals.append(float(v))
        rhs = float(vec_rhs[i])
        push_row(cols, vals, rhs, rhs)

# (A) Dynamics:
# k=1:  x1 - B*u1 = A*x0
# k>=2: xk - A*x_{k-1} - B*uk = 0
Ax0 = A @ x0
add_vec_eq([(ix_x(1).start, np.eye(nx)), (ix_u(1).start, -B)], Ax0.reshape((nx, 1)))
for k in range(2, N+1):
    blk = [
        (ix_x(k).start,     np.eye(nx)),
        (ix_x(k-1).start,  -A),
        (ix_u(k).start,    -B),
    ]
    add_vec_eq(blk, np.zeros((nx, 1)))

# (B) |Qx| via t_x:  t_x - Qx >= 0  and  t_x + Qx >= 0
for k in range(1, N+1):
    for i in range(nx):
        # t_x(i) - (Qx)(i) >= 0
        cols = [ix_tx(k).start + i]
        vals = [1.0]
        for j in range(nx):
            qij = Q[i, j]
            if qij != 0.0:
                cols.append(ix_x(k).start + j)
                vals.append(-qij)
        push_row(cols, vals, 0.0, np.inf)
        # t_x(i) + (Qx)(i) >= 0
        cols = [ix_tx(k).start + i]
        vals = [1.0]
        for j in range(nx):
            qij = Q[i, j]
            if qij != 0.0:
                cols.append(ix_x(k).start + j)
                vals.append(+qij)
        push_row(cols, vals, 0.0, np.inf)

# (C) |u| via t_u:  t_u - u >= 0,  t_u + u >= 0
for k in range(1, N+1):
    push_row([ix_tu(k).start + 0, ix_u(k).start + 0], [1.0, -1.0], 0.0, np.inf)
    push_row([ix_tu(k).start + 0, ix_u(k).start + 0], [1.0, +1.0], 0.0, np.inf)

# Stack constraint matrix and bounds (numeric → DM)
A_mat = ca.DM(np.vstack([np.array(r) for r in rows])) if rows else ca.DM.zeros(0, n)
lba = ca.DM(np.array(lba_list, dtype=float).reshape(-1, 1))
uba = ca.DM(np.array(uba_list, dtype=float).reshape(-1, 1))
lbx = ca.DM(lbx_num)
ubx = ca.DM(ubx_num)
g_lin_dm = ca.DM(g_lin)

# ---------------- Build NLP with SX decision vector ----------------
z = ca.SX.sym("z", n)

# Objective (linear): 0.5*zᵀ·0·z + g_linᵀ z  == g_linᵀ z
f_expr = ca.dot(g_lin_dm, z)

# Linear constraints: g_expr = A_mat * z
g_expr = A_mat @ z

nlp = {"x": z, "f": f_expr, "g": g_expr}

# Quiet IPOPT options
opts = {
    "print_time": True,
    "ipopt": {
        "print_level": 0,   # 0 = none
        "sb": "yes",        # silent barrier messages
        # Optional tweaks:
        # "tol": 1e-9,
        # "linear_solver": "mumps",
    }
}

solver = ca.nlpsol("solver", "ipopt", nlp, opts)

# Solve (no initial guess required; provide bounds)
sol = solver(x0=ca.DM.zeros(n, 1), lbg=lba, ubg=uba, lbx=lbx, ubx=ubx)

# ---------------- Unpack solution ----------------
z_opt = np.array(sol["x"]).reshape(-1)

X = np.zeros((N, nx))
U = np.zeros((N, nu))
for k in range(1, N+1):
    X[k-1] = z_opt[ix_x(k)]
    U[k-1] = z_opt[ix_u(k)]

print("Solver: IPOPT via CasADi")
print("U* =", np.round(U.ravel(), 3))
print("X* =\n", np.round(X, 3))