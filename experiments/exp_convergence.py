"""
Convergence and accuracy (RQ2) with offline hyperparameter tuning.

Same setting as the default experiment (natural partition, random for Housing; 50-110 new
samples per node and round; 30 rounds; 10 runs), but every method gets its best
hyperparameters, selected OFFLINE on the global validation set (the test set is never used):

  1. Ridge penalty lambda of Cent and FedDelta (the same value for both, so that FedDelta
     still reproduces Cent): closed-form ridge on all the training data of the first run's
     split, best validation R² over LAMBDA_GRID. The gradient-based methods keep their own
     lambda_reg from the YAML: their local objective scales the penalty with the mini-batch,
     so the same value would over-regularize them.
  2. The other baselines (*_grid keys in config_experiment.yaml): learning rate, mu and
     server learning rate of the gradient-based methods, and n_estimators / max_depth of
     CentRF and the learning rate of CentMLP. Grid search with a short simulation (federated
     for the federated methods, centralized on the pooled samples for CentRF/CentMLP),
     scored on validation (TUNE = "on"; see src/simulation/core/tuning.py).

This tuning uses data that would not be available in an incremental deployment, so it is a
best case for the baselines.
All parameters are set at the top of main().

Outputs (results/convergence): table.csv (final R², MSE, time per round), lambda.csv
(validation R² of every lambda), per-round metrics under framework/<tag>/ (for
plot_convergence.py), tuned values of the baselines in framework/<tag>/.../tuning/.
"""
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

import common as C
from src.simulation.settings.data import load_raw_data


def tune_lambda(ds: str, grid, num_nodes: int):
    """Validation R² of ridge (unpenalized intercept) on all training data, for each lambda."""
    cfg = {**C.base_config(), "num_nodes": num_nodes, "global_split": False}
    node_data, *_ = load_raw_data(ds, C.DATASETS[ds]["target"], cfg)
    stack = lambda s: (np.vstack([v[s][0] for v in node_data.values() if len(v[s][1])]),
                       np.concatenate([v[s][1] for v in node_data.values() if len(v[s][1])]))
    (Xt, yt), (Xv, yv) = stack("train"), stack("val")
    Xt1 = np.hstack([np.ones((len(Xt), 1)), Xt])
    Xv1 = np.hstack([np.ones((len(Xv), 1)), Xv])
    A, B = Xt1.T @ Xt1, Xt1.T @ yt
    D = np.eye(A.shape[0]); D[0, 0] = 0.0                       # intercept not penalized
    rows = []
    for lam in grid:
        w = np.linalg.solve(A + lam * D, B)
        r2 = 1 - np.sum((Xv1 @ w - yv) ** 2) / np.sum((yv - yv.mean()) ** 2)
        rows.append({"Dataset": C.DATASETS[ds]["label"], "lambda": lam, "val R2": r2})
    best = max(rows, key=lambda r: r["val R2"])["lambda"]
    return best, rows


def main():
    # ---- parameters ------------------------------------------------------
    DATASETS = ["housing", "discharges", "european_survey_adult", "length_of_stay"]
    ALGS = [C.CENT, C.FEDDELTA, "CentRF", "CentMLP",
            "FedGD (FedAvg)", "FedGD (FedNova)", "FedGD (FedProx)", "FedSCAFFOLD",
            "FedGD (FedAvgM)", "FedMLP (FedAvg)"]
    NUM_NODES = 10                  # natural nodes are used when the dataset has groups
    N_RUNS = 1                     # independent runs
    N_ROUNDS = 30                   # rounds per run (fewer if the data run out)
    SAMPLES_MIN = 50                # new samples per node and round (min)
    SAMPLES_MAX = 110               # new samples per node and round (max)
    TUNE = "on"                     # "on": grid search of the baselines | "yaml": fixed values
    TUNE_LAMBDA = True              # select the ridge lambda on validation
    LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    LAMBDA_FIXED = 0.01             # used when TUNE_LAMBDA is False
    FORCE = False                   # True: recompute even if cached
    # ----------------------------------------------------------------------
    logging.getLogger().setLevel(logging.ERROR)
    run_args = SimpleNamespace(runs=N_RUNS, rounds=N_ROUNDS, tune=TUNE, quick=False)
    samples = [{"node_min_samples": SAMPLES_MIN, "node_max_samples": SAMPLES_MAX,
                "min_samples_train": SAMPLES_MIN}]
    edir = C.exp_dir("convergence")

    rows, lam_rows = [], []
    for ds in DATASETS:
        if TUNE_LAMBDA:
            lam, lr = tune_lambda(ds, LAMBDA_GRID, NUM_NODES)
            lam_rows += lr
            print(f"convergence {ds}: lambda = {lam:g} (best validation R²)")
        else:
            lam = LAMBDA_FIXED

        cfg = C.make_config(run_args, [ds], ALGS, num_nodes=NUM_NODES, samples_mode="fixed",
                            node_samples_list=samples, global_split=False,
                            lambda_scope="ridge")
        cfg["algorithms"][C.FEDDELTA]["lambda_reg"] = [lam]   # applied to every linear model
        tag = C.tag(run_args, ds, f"lam{lam:g}", f"b{SAMPLES_MIN}-{SAMPLES_MAX}")
        hist = C.histories_of(C.run(cfg, tag, edir, FORCE), ds)

        for a in ALGS:
            r2, mse = C.final_values(hist, a, "r2"), C.final_values(hist, a, "mse")
            t = C.curves(hist, a, "exec_time")
            t_run = np.nanmean(t, axis=1) if t.size else np.array([np.nan])
            rows.append({"Dataset": C.DATASETS[ds]["label"], "Algorithm": C.short(a),
                         "lambda": lam, "Rounds": t.shape[1] if t.size else np.nan,
                         "R2 mean": np.nanmean(r2), "R2 sd": np.nanstd(r2, ddof=1),
                         "MSE mean": np.nanmean(mse), "MSE sd": np.nanstd(mse, ddof=1),
                         "Time/round (s)": np.nanmean(t_run),
                         "Time/round sd (s)": np.nanstd(t_run, ddof=1)})

    C.save_table(pd.DataFrame(rows), edir / "table")
    if lam_rows:
        C.save_table(pd.DataFrame(lam_rows), edir / "lambda")


if __name__ == "__main__":
    main()
