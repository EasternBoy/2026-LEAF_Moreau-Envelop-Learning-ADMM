
import pickle
import numpy as np
import cvxpy as cp
import os
import jax.numpy as jnp
import jax
from lpcf.pcf import PCF

seed = 0
np.random.seed(seed)

data = np.load(os.path.join("..","data","train_rho=10.npz"))

X     = data["input"].T
Y     = data["enve"].reshape(-1, 1)
grad  = data["grad"]
Theta = np.ones(len(Y))

n = X.shape[1]


pcf = PCF(activation='logistic')
stats = pcf.fit(Y, X, Theta, cores=10)

print(f'Elapsed time: {stats['time']} s')
print(f'R2 score: {stats['R2']}')

# export to jax

f = pcf.tojax()

# evaluate

data = np.load(os.path.join("..","data","test_rho=10.npz"))

X_test     = data["input"].T
Y_test     = data["enve"].reshape(-1, 1)
grad_test  = data["grad"]
Theta_test      = np.ones(len(Y_test))


Y_hat = f(X_test, Theta_test)

print(max(np.abs(Y_hat - Y_test)))

def fn(x, t):
    preds = f(x,t)              # shape (1,1)
    return jnp.squeeze(preds)   # shape () scalar

f_grad = jax.grad(fn)