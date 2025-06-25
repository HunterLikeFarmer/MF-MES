#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scripts.MultiFidelityBO import multifidelity_bayesian_opt as MFBO  # import MF-MES core :contentReference[oaicite:0]{index=0}
from scripts.myutils import myutils                               # for initial design :contentReference[oaicite:1]{index=1}

# -- 1) Define a simple test function with two fidelities --
#    True underlying function we want to minimize:
def f_high(x):
    return (x - 2.0)**2

#    A cheap, biased version:
def f_low(x):
    return f_high(x) + 1.0  + 0.2 * np.random.randn(*x.shape)

# For MF-MES (which seeks to MAXIMIZE its inputs), we negate:
def g_high(x):
    return - f_high(x)

def g_low(x):
    return - f_low(x)

# -- 2) Problem setup -----------------------------------------------
# 1-D domain [0, 4]
input_dim = 1
bounds = np.array([[0.0], [4.0]])
interval = bounds[1] - bounds[0]

# initial design: 5 points at low fidelity, 4 at high fidelity
FIRST_NUM = [5 * input_dim, 4 * input_dim]
# draw those via Latin Hypercube
X_list = myutils.initial_design(FIRST_NUM, input_dim, bounds)  # list of two arrays :contentReference[oaicite:2]{index=2}
print(X_list)
# evaluate them
Y_low  = g_low(X_list[0])
Y_high = g_high(X_list[1])
Y_list = [Y_low, Y_high]

# how many points we've evaluated at each fidelity so far
eval_num = FIRST_NUM.copy()

# GP hyper-parameter bounds
kernel_bounds = np.array([interval/10.0, interval]).reshape(2, input_dim)

# relative query costs (low is cheap, high is expensive)
cost = np.array([1.0, 5.0])

# -- 3) Instantiate the MF-MES optimizer ----------------------------
optimizer = MFBO.MultiFidelityMaxvalueEntropySearch(
    X_list        = X_list,
    Y_list        = Y_list,
    eval_num      = eval_num,
    bounds        = bounds,
    kernel_bounds = kernel_bounds,
    M             = 2,
    cost          = cost,
    sampling_num      = 10,
    sampling_method   = 'RFM',
    model_name     = 'MFGP',
    optimize       = True
)

# -- 4) Run a few sequential iterations -----------------------------
MAX_ITERS = 20
for i in range(MAX_ITERS):
    # get next (x*, fidelity) to evaluate
    nxt = optimizer.next_input()                      # returns [[x_opt, m_opt]] :contentReference[oaicite:4]{index=4}
    x_new, m_new = nxt[0, :-1], int(nxt[0, -1])
    # evaluate the selected fidelity
    if m_new == 0:
        y_new = g_low (x_new[None,:])
    else:
        y_new = g_high(x_new[None,:])

    # prepare lists for update:
    add_X = [ np.empty((0, input_dim)), np.empty((0, input_dim)) ]
    add_Y = [ np.empty((0, 1)),         np.empty((0, 1))         ]
    add_X[m_new] = x_new[None,:]
    add_Y[m_new] = y_new.reshape(-1,1)

    # feed it back to the GP—and re-sample maxima
    optimizer.update(add_X, add_Y, optimize=True)

    print(f"Iter {i:2d}: eval @ x={x_new.ravel()[0]:.3f}, fidelity={m_new}, "
          f"f(x)={( -y_new ).ravel()[0]:.3f}")

# -- 5) Report final minimizer --------------------------------------
x_star, g_star = optimizer.posteriori_maximum()  # maximizes g = -f  :contentReference[oaicite:5]{index=5}
x_min = x_star.ravel()[0]
f_min = - g_star
print("\nEstimated minimizer: x* =", x_min, "with f(x*) ≈", f_min)
