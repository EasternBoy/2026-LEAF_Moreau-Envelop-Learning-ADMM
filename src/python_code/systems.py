import numpy as np

def LTI_2D():
    A = np.array([[2.0, -1.0],
                  [1.0,  0.2]], dtype=float)
    B = np.array([[1.0],
                  [0.0]], dtype=float)
    nx = 2
    nu = 1

    Q = np.eye(nx)
    R = np.array([[2.0]])

    N = 7
    x_min = np.full(nx, -5.0)
    x_max = np.full(nx, +5.0)
    u_min = np.full(nu, -1.0)
    u_max = np.full(nu, +1.0)

    x0 = np.array([3., 1.])

    return dict(A = A, B = B, nx = nx, nu = nu, Q = Q, R = R,  N = N,  x_max = x_max,  x_min = x_min,  u_min = u_min,  u_max  = u_max, x0 = x0)

def LTI_1D():
    # --- Model data ---
    A = 1.
    B = 1.

    nx = 1           # number of states

    nu = 1           # number of inputs

    # --- MPC data ---
    Q = 1.        # 2×2 identity
    R = 2.          # scalar
    N = 10            # horizon

    x_max =  5.
    x_min = -5.
    u_max =  1.
    u_min = -1. 

    return dict(A = A, B = B, nx = nx, nu = nu, Q = Q, R = R,  N = N,  x_max = x_max,  x_min = x_min,  u_min = u_min,  u_max  = u_max)