import numpy as np
import casadi as ca

GUROBI_VERSION=120 


print("CasADi:", getattr(ca, "__version__", "unknown"))


# ---------------- Problem data ----------------
A = np.array([[2.0, -1.0],
              [1.0,  0.2]])
B = np.array([[1.0],
              [0.0]])
nx, nu = 2, 1

Q = np.eye(2)     # used inside ||Q x_k||_1
R = 2.0
N = 7

x0 = np.array([3.0, 1.0])

x_min, x_max = -5.0, 5.0
u_min, u_max = -1.0, 1.0

# ---------------- Variable stacking ----------------
# z = [x_1..x_N, u_1..u_N, t_x_1..t_x_N, t_u_1..t_u_N]
nx_blk = nx*N
nu_blk = nu*N
ntx_blk = nx*N
ntu_blk = nu*N
n = nx_blk + nu_blk + ntx_blk + ntu_blk

def ix_x(k):   # 1..N → slice
    s = (k-1)*nx
    return slice(s, s+nx)

def ix_u(k):
    s = nx_blk + (k-1)*nu
    return slice(s, s+nu)

def ix_tx(k):
    s = nx_blk + nu_blk + (k-1)*nx
    return slice(s, s+nx)

def ix_tu(k):
    s = nx_blk + nu_blk + ntx_blk + (k-1)*nu
    return slice(s, s+nu)


# ---------------- QP in CasADi low-level format ----------------
# Minimize 0.5 z^T H z + g^T z
# Subject to:
#   lbx <= z <= ubx                             (variable bounds)
#   lba <= A_mat @ z <= uba                    (linear (in)equality)

# Objective: L1 via auxiliaries → H = 0, linear cost on t_x and t_u
H = ca.DM.zeros(n, n)                # sparse pattern used below
g = ca.DM.zeros(n, 1)
for k in range(1, N+1):
    g[ix_tx(k)] = 1.0                # sum_i t_x
    g[ix_tu(k)] = R                  # R * t_u

# Variable bounds
lbx = -ca.inf*ca.DM.ones(n, 1)
ubx =  ca.inf*ca.DM.ones(n, 1)

# x bounds
for k in range(1, N+1):
    lbx[ix_x(k)] = x_min
    ubx[ix_x(k)] = x_max

# u bounds
for k in range(1, N+1):
    lbx[ix_u(k)] = u_min
    ubx[ix_u(k)] = u_max

# t_x, t_u >= 0
for k in range(1, N+1):
    lbx[ix_tx(k)] = 0.0
    lbx[ix_tu(k)] = 0.0

# Build linear constraints rows into A_rows, lba, uba
rows = []
lba = []
uba = []

def push_row(cols, vals, low, up):
    # one row: sum_j vals[j] * z[cols[j]] in [low, up]
    nz = len(cols)
    r = ca.SX.zeros(1, n)
    for j in range(nz):
        r[0, cols[j]] = vals[j]
    rows.append(ca.DM(r))
    lba.append(low)
    uba.append(up)

# Helper to add vector equalities compactly:
def add_vec_eq(col_blocks, vec_rhs):
    # enforces: sum_j M_j * z[col_block_j] = rhs (componentwise)
    # Here we fill rows one-by-one.
    for i in range(vec_rhs.shape[0]):
        cols = []
        vals = []
        # accumulate entries from each block matrix row i
        for (block_start, M) in col_blocks:
            for j in range(M.shape[1]):
                v = M[i, j]
                if abs(v) > 0:
                    cols.append(block_start + j)
                    vals.append(float(v))
        push_row(cols, vals, float(vec_rhs[i]), float(vec_rhs[i]))

# (A) Dynamics:
# k=1:  x1 - B u1 = A x0
# k>=2: xk - A x_{k-1} - B uk = 0
A_dm = ca.DM(A); B_dm = ca.DM(B)
Ax0 = A @ x0

# k=1
# [I on x1] + [-B on u1] == Ax0
add_vec_eq([(ix_x(1).start, np.eye(nx)), (ix_u(1).start, -B)], Ax0.reshape((nx,1)))

# k>=2
for k in range(2, N+1):
    # [I on xk] + [-A on x_{k-1}] + [-B on uk] == 0
    blk = [
        (ix_x(k).start,      np.eye(nx)),
        (ix_x(k-1).start,   -A),
        (ix_u(k).start,     -B),
    ]
    add_vec_eq(blk, np.zeros((nx,1)))

# (B) |Qx| via t_x:
#   t_x >=  Q x  →  (I on t_x) + (-Q on x) >= 0
#   t_x >= -Q x  →  (I on t_x) + (+Q on x) >= 0
Q_dm = ca.DM(Q)
for k in range(1, N+1):
    # tx - Q x >= 0
    for i in range(nx):
        cols = [ix_tx(k).start + i]
        vals = [1.0]
        for j in range(nx):
            qij = Q[i, j]
            if qij != 0.0:
                cols.append(ix_x(k).start + j)
                vals.append(-qij)
        push_row(cols, vals, 0.0, ca.inf)
    # tx + Q x >= 0
    for i in range(nx):
        cols = [ix_tx(k).start + i]
        vals = [1.0]
        for j in range(nx):
            qij = Q[i, j]
            if qij != 0.0:
                cols.append(ix_x(k).start + j)
                vals.append(+qij)
        push_row(cols, vals, 0.0, ca.inf)

# (C) |u| via t_u:
#   t_u - u >= 0 ; t_u + u >= 0
for k in range(1, N+1):
    # t_u - u >= 0
    push_row([ix_tu(k).start+0, ix_u(k).start+0], [1.0, -1.0], 0.0, ca.inf)
    # t_u + u >= 0
    push_row([ix_tu(k).start+0, ix_u(k).start+0], [1.0, +1.0], 0.0, ca.inf)

# Stack A and bounds
A_mat = ca.densify(ca.vertcat(*rows)) if rows else ca.DM.zeros(0, n)
lba = ca.DM(lba)
uba = ca.DM(uba)

print("------------------------------------------------------------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------------------------------------------------------")
# ---------------- Solve with CasADi ----------------
qp = {
    "h": H.sparsity(),   # low-level API (docs §4.6.2)
    "a": A_mat.sparsity()
}
# Try proxQP first; fallback to OSQP/qpOASES if not built-in
for plugin in ["proxqp", "osqp", "qpoases"]:
    options = {"print_problem": False, 
               "print_time": True, 
               "verbose": False,
               "print_out":False,
               "print_in":False}
    solver = ca.conic("S", plugin, qp, options)  # solver creation

    print("-------------------------")
    print("-------------------------")
    sol = solver(h=H, g=g, a=A_mat, lba=lba, uba=uba, lbx=lbx, ubx=ubx)

    st = solver.stats()      # a Python dict
    print(st.keys())         # see what your plugin exposes
    print("Total processing time: ", st.get("t_proc_total")*1000, "miliseconds")     # CPU time (s), if present

    z = np.array(sol["x"]).reshape(-1)

    # Extract X, U
    X = np.zeros((N, nx))
    U = np.zeros((N, nu))
    for k in range(1, N+1):
        X[k-1] = z[ix_x(k)]
        U[k-1] = z[ix_u(k)]

    print("CasADi-QP solver:", plugin)
    print("U*  =", np.round(U.ravel(),3))