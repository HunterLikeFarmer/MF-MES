#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import argparse

# your CSV-loading helpers
from data import (
    get_pair_low_fidelity,
    get_wall_clock_times_low_fidelity,
    get_pair_high_fidelity,
    get_wall_clock_times_high_fidelity,
)

# MF-MES core & utils
from scripts.MultiFidelityBO import multifidelity_bayesian_opt as MFBO
from scripts.myutils import myutils

parser = argparse.ArgumentParser(
        description="Plot iteration vs. Wall Clock Time from JSON logs and a CSV cost file."
    )

parser.add_argument(
        "--output",
        default="zresults/cluster_cost.json",
        help="output of the result"
    )

# -- 1) Load and prepare data ----------------------------------------
X_low_all   = np.array(get_pair_low_fidelity())            
Y_low_all   = np.array(get_wall_clock_times_low_fidelity())
X_high_all  = np.array(get_pair_high_fidelity())           
Y_high_all  = np.array(get_wall_clock_times_high_fidelity())

# column-vectors
Y_low_all   = Y_low_all.reshape(-1,1)
Y_high_all  = Y_high_all.reshape(-1,1)

# keep a copy of the true (positive) high-fidelity times for tracking
Y_high_orig = Y_high_all.copy()

# negate for MF-MES (it maximizes)
Y_low_all  = -Y_low_all
Y_high_all = -Y_high_all

# dictionary to look up the true high-fidelity time by x
orig_high = { tuple(x): float(y) 
              for x,y in zip(X_high_all, Y_high_all.flatten()) }

# -- 2) Build mutable “pools” (with popping) ------------------------
pool_X = [X_low_all.copy(),  X_high_all.copy()]
pool_Y = [Y_low_all.copy(),  Y_high_all.copy()]

# -- 3) GP + domain setup -------------------------------------------
d = X_high_all.shape[1]       # dimensionality
lower = np.vstack(pool_X).min(axis=0)
upper = np.vstack(pool_X).max(axis=0)
bounds = np.vstack((lower, upper))   # continuous search box

span = upper - lower
kernel_bounds = np.vstack((span/10.0, span))

cost = np.array([1.0, 1.2])   # low vs high fidelity cost

# -- 4) Initial seed via LHS ----------------------------------------
init_nums = [5*d, 4*d]
X0, X1    = myutils.initial_design(init_nums, d, bounds)

def find_and_pop(x_cont, m):
    """
    Find the nearest unseen discrete point in pool_X[m],
    pop it from the pool, and return its x (1×d) and y (1×1).
    """
    arr = pool_X[m]
    dists = np.linalg.norm(arr - x_cont, axis=1)
    idx  = np.argmin(dists)
    # grab
    x_disc = arr[idx:idx+1]             # shape (1,d)
    y_disc = pool_Y[m][idx:idx+1]       # shape (1,1)
    # remove from pool
    pool_X[m] = np.delete(pool_X[m], idx, axis=0)
    pool_Y[m] = np.delete(pool_Y[m], idx, axis=0)
    return x_disc, y_disc

# build the initial X_list, Y_list
X_list, Y_list = [], []
for m, Xinit in enumerate((X0, X1)):
    xs, ys = [], []
    for x_cont in Xinit:
        x_d, y_d = find_and_pop(x_cont, m)
        xs.append(x_d)
        ys.append(y_d)
    X_list.append(np.vstack(xs))
    Y_list.append(np.vstack(ys))

eval_num = [X_list[0].shape[0], X_list[1].shape[0]]

# -- 5) Instantiate MF-MES optimizer -------------------------------
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
    model_name        = 'MFGP',
    optimize          = True
)

# -- 6) Sequential loop under cost budget ---------------------------
max_cost  = 100.0
print("cost[0] is ")
print(cost[0])
print("cost[1] is ")
print(cost[1])
print("eval_num[0] is ")
print(eval_num[0])
print("eval_num[1] is ")
print(eval_num[1])
cum_cost  = cost[0]*eval_num[0] + cost[1]*eval_num[1]
best_hi   = -float('inf')
best_x    = None
iteration = 0
MAX_ITERS = 50

# prepare JSON log
log = []

while cum_cost < max_cost and iteration < MAX_ITERS:
    iteration += 1

    # (a) propose continuous x and fidelity m
    proposal = optimizer.next_input()[0]
    x_cont, m  = proposal[:-1], int(proposal[-1])

    # (b) round to nearest discrete and pop it
    x_disc, y_disc = find_and_pop(x_cont, m)

    # (c) correctly prepare add_X/add_Y for update
    add_X = [ np.empty((0,d)), np.empty((0,d)) ]
    add_Y = [ np.empty((0,1)), np.empty((0,1)) ]
    add_X[m] = x_disc
    add_Y[m] = y_disc

    optimizer.update(add_X, add_Y, optimize=True)

    # (d) fetch the *true* high-fidelity value and update best
    x_key     = tuple(x_disc.ravel())
    y_hi_true = orig_high[x_key]
    if y_hi_true > best_hi:
        best_hi = y_hi_true
        best_x  = x_disc.ravel().copy()

    # (e) update cost
    cum_cost += cost[m]

    # record JSON entry
    entry = {
        "iteration": iteration,
        "current": {
            "A100": float(x_disc[0,0]),
            "V100": float(x_disc[0,1])
        },
        "best_so_far": {
            "A100": float(best_x[0]),
            "V100": float(best_x[1]),
            "wall_clock_time": -best_hi
        },
        "cumulative_cost": cum_cost
    }
    log.append(entry)

    print(f"Iter {iteration:2d}: m={m}, x={x_disc.ravel()}, "
          f"Best_x={best_x}, MF_y={-y_disc[0,0]:.4f}, "
          f"true_hi={-y_hi_true:.4f}, best_hi={-best_hi:.4f}, "
          f"cum_cost={cum_cost:.1f}")

# -- 7) Final report & write JSON log -------------------------------
print("\n=== Final Result ===")
print(f"Best x         = {best_x}")
print(f"Min Wall-Clock = {best_hi:.4f}")
args = parser.parse_args()
with open(args.output, "w") as f:
    json.dump(log, f, indent=2)
