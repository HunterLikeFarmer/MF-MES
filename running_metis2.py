#!/usr/bin/env python3
# run_mfmes.py
# Example: python run_mfmes.py dataset.npz 10 5 1000

import sys
import numpy as np

# MF-MES implementation
from myBO.scripts.MultiFidelityBO.multifidelity_bayesian_opt import MultiFidelityMaxvalueEntropySearch
# Utilities for LHS etc. (only used to set up kernel bounds here)
from myBO.scripts.myutils import myutils

def main(dataset_path, init_low, init_high, budget_max):
    # ─────── 1. Load your dataset ───────
    # Expects a .npz with keys: X_low, Y_low, X_high, Y_high, cost_low, cost_high
    data = np.load(dataset_path)
    X_low, Y_low = data['X_low'], data['Y_low']
    X_high, Y_high = data['X_high'], data['Y_high']
    cost_low, cost_high = float(data['cost_low']), float(data['cost_high'])

    # Discrete pools
    pool_X = [X_low.copy(), X_high.copy()]
    pool_Y = [Y_low.copy(), Y_high.copy()]
    costs  = [cost_low, cost_high]
    M = 2
    d = X_low.shape[1]

    # ─────── 2. Bounds & kernel bounds ───────
    # bounds: shape (2, d) so that bounds.T is [(min, max)...] per dim
    allX = np.vstack((X_low, X_high))
    bounds = np.vstack((allX.min(axis=0), allX.max(axis=0)))
    interval = bounds[1] - bounds[0]
    # per‐dimension lengthscale bounds: [interval/10, interval]
    kernel_bounds = np.array([interval/10.0, interval]).astype(float)

    # ─────── 3. Initial design ───────
    np.random.seed(0)
    # pick init_low distinct low‐fidelity points, init_high high‐fidelity
    def init_sample(pool_Xm, pool_Ym, n):
        n = min(n, pool_Xm.shape[0])
        idx = np.random.choice(pool_Xm.shape[0], n, replace=False)
        return idx, pool_Xm[idx], pool_Ym[idx]
    idx0, X0_low,  Y0_low  = init_sample(pool_X[0], pool_Y[0], init_low)
    idx1, X0_high, Y0_high = init_sample(pool_X[1], pool_Y[1], init_high)
    # remove those from the pool
    mask0 = np.ones(pool_X[0].shape[0], bool); mask0[idx0] = False
    mask1 = np.ones(pool_X[1].shape[0], bool); mask1[idx1] = False
    pool_X[0], pool_Y[0] = pool_X[0][mask0], pool_Y[0][mask0]
    pool_X[1], pool_Y[1] = pool_X[1][mask1], pool_Y[1][mask1]

    X_list = [X0_low, X0_high]
    Y_list = [Y0_low, Y0_high]
    eval_num = [X0_low.shape[0], X0_high.shape[0]]

    # Track true best high‐fidelity y so far
    best_y = np.min(Y0_high)
    best_x = X0_high[np.argmin(Y0_high)]
    best_f = 1  # fidelity index

    # initial cost
    current_cost = eval_num[0]*costs[0] + eval_num[1]*costs[1]

    # ─────── 4. Instantiate MF-MES ───────
    optimizer = MultiFidelityMaxvalueEntropySearch(
        X_list=X_list,
        Y_list=Y_list,
        eval_num=eval_num,
        bounds=bounds,
        kernel_bounds=kernel_bounds,
        M=M,
        cost=np.array(costs),
        sampling_num=10,            # # of f* samples
        sampling_method='RFM',      # or 'Gumbel'
        pool_X=pool_X,
        optimize=True
    )

    print(f"{'Iter':>4s}  {'Cost':>8s}   {'x':>12s}  {'f'}  {'y_sel':>8s}   {'best_y':>8s}")
    print("-"*60)

    iteration = 0
    # ─────── 5. BO loop ───────
    while current_cost < budget_max and any(len(px)>0 for px in pool_X):
        # 5a) Select next point (sequential)
        x_sel, pool_X = optimizer.next_input_pool(pool_X)
        x, m = x_sel[0,:-1], int(x_sel[0,-1])

        # 5b) Lookup true y
        # find the matching index in the original pool_Y[m]
        # (we removed sampled ones from pool_X, so use pool_Y[m] directly)
        # but simplest: match against X_list[m] + recent x_sel
        # so we keep separate orig arrays:
        if m==0:
            # for low
            y_sel = pool_Y[0][np.all(pool_X[0]==x,axis=1)][0]
        else:
            y_sel = pool_Y[1][np.all(pool_X[1]==x,axis=1)][0]

        # 5c) Update cost & best
        current_cost += costs[m]
        if m==1 and y_sel < best_y:
            best_y, best_x, best_f = y_sel, x, m

        # 5d) Append to training data & update GP
        new_X = [[],[]]
        new_Y = [[],[]]
        new_X[m] = x.reshape(1,-1)
        new_Y[m] = np.array([y_sel])
        optimizer.update(new_X, new_Y, optimize=False)

        iteration += 1
        print(f"{iteration:4d}  {current_cost:8.2f}  {x!s:12s}  {m:d}  {y_sel:8.4g}  {best_y:8.4g}")

    print("\nDone. Best high‐fidelity y = {:.6g} at x = {} (fidelity={}).".format(best_y, best_x.tolist(), best_f))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_mfmes.py <dataset.npz> <init_low> <init_high> <budget_max>")
        sys.exit(1)
    _, ds, il, ih, bm = sys.argv
    main(ds, int(il), int(ih), float(bm))
