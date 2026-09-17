import numpy as np

def lti_parameter():
    A = np.array([[2.0, -1.0],
                  [1.0,  0.2]], dtype=float)
    B = np.array([[1.0],
                  [0.0]], dtype=float)
    nx = 2
    nu = 1
    Q = np.eye(nx)
    R = np.array([[2.0]])
    N = 50
    x_min = np.full(nx, -5.0)
    x_max = np.full(nx, +5.0)
    u_min = np.full(nu, -1.0)
    u_max = np.full(nu, +1.0)
    x0 = np.array([3., 1.])
    return {
        "A": A, "B": B, "nx": nx, "nu": nu,
        "Q": Q, "R": R, "N": N,
        "x_max": x_max, "x_min": x_min,
        "u_min": u_min, "u_max": u_max,
        "x0": x0
    }