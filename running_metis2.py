#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# your helpers for reading the CSV
from data import (
    get_pair_low_fidelity,
    get_wall_clock_times_low_fidelity,
    get_pair_high_fidelity,
    get_wall_clock_times_high_fidelity,
)

# MF-MES core
from scripts.MultiFidelityBO import multifidelity_bayesian_opt as MFBO
from scripts.myutils import myutils

# -- 1) Load both fidelities from CSV ------------------------------
# (you wrote these already)
X_low  = np.array(get_pair_low_fidelity())    # shape (n₀, d)
Y_low  = np.array(get_wall_clock_times_low_fidelity()).reshape(-1,1)
X_high = np.array(get_pair_high_fidelity())   # shape (n₁, d)
Y_high = np.array(get_wall_clock_times_high_fidelity()).reshape(-1,1)

# -- 2) Problem setup -----------------------------------------------
# dimensionality
input_dim = X_high.shape[1]    # here d = 2 for (A100, V100)

# pack into lists
X_list  = [X_low, X_high]
Y_list  = [Y_low, Y_high]
eval_num = [len(X_low), len(X_high)]   # [n₀, n₁]

# compute domain‐bounds from the union of points
all_X = np.vstack(X_list)
lower = all_X.min(axis=0)
upper = all_X.max(axis=0)
bounds = np.vstack((lower, upper))     # shape (2, d)

# lengthscale bounds: [ (upper−lower)/10 , (upper−lower) ]
span = upper - lower
kernel_bounds = np.vstack((span/10.0, span))  # shape (2, d)

# relative cost of each fidelity (you can tune these)
cost = np.array([1.0, 5.0])

# -- 3) Instantiate the MF-MES optimizer ----------------------------
optimizer = MFBO.MultiFidelityMaxvalueEntropySearch(
    X_list        = X_list,
    Y_list        = Y_list,
    eval_num      = eval_num,
    bounds        = bounds,
    kernel_bounds = kernel_bounds,
    M             = 2,               # two fidelities
    cost          = cost,
    sampling_num      = 10,
    sampling_method   = 'RFM',
    model_name        = 'MFGP',
    optimize          = True
)

# -- 4) Estimate the minimizer via the GP ---------------------------
# MF-MES is built to MAXIMIZE g = –f, so posteriori_maximum() finds
#   x* = argmax_x (–WallTime), i.e. argmin_x WallTime
x_star, g_star = optimizer.posteriori_maximum()

# unpack
x_opt = x_star.ravel()   # [A100_opt, V100_opt]
y_opt = -g_star          # predicted minimum Wall Clock Time

print(f"Estimated optimal configuration:")
print(f"  A100 = {x_opt[0]:.4f}, V100 = {x_opt[1]:.4f}")
print(f"  Predicted min Wall Clock Time = {y_opt:.4f}")
