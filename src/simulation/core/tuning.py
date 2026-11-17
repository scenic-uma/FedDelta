"""
Grid search for the baselines (federated FedGD/FedSCAFFOLD/FedMLP and centralized CentRF/CentMLP).

Any YAML key ending in `_grid` (e.g. eta0_grid, mu_grid) is searched.
Each candidate runs a short federated simulation on the nodes' training
data and is scored on the global VALIDATION set (test set is never used).
The best values are cached in config.tuned_params and reused by later runs.
"""
import inspect
import itertools
import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.simulation.config import RESULTS_DIR, ExperimentConfig
from src.simulation.settings.algorithms import get_algorithm_instance

logger = logging.getLogger(__name__)


def _grid_keys(params: Dict) -> Dict[str, list]:
    return {k[:-5]: v for k, v in params.items() if k.endswith("_grid") and isinstance(v, list)}


def _is_finite(alg) -> bool:
    """True if the global model has no NaN/inf (diverged candidates are dropped early)."""
    w = alg.global_model
    ws = w.get_weights() if hasattr(w, "get_weights") else [np.asarray(w, dtype=float)]
    return all(np.all(np.isfinite(x)) for x in ws)


async def _simulate(alg, nodes, config: ExperimentConfig, rounds: int, rng) -> None:
    """Short FedAvg-style simulation; node state is not modified."""
    for _ in range(rounds):
        for node in nodes:
            n_avail = len(node.X_full)
            if n_avail == 0:
                continue
            n = int(rng.integers(config.node_min_samples, config.node_max_samples + 1))
            idx = rng.choice(n_avail, size=min(n, n_avail), replace=False)
            args = (node.X_full[idx], node.y_full[idx].reshape(-1, 1),
                    config.node_epochs, config.node_batch_size)
            if "node_id" in inspect.signature(alg.fit).parameters:
                update = alg.fit(*args, node_id=node.id)
            else:
                update = alg.fit(*args)
            await alg.aggregate_update(update)
        alg.complete_aggregation_round()
        alg.reset_elapsed_metrics()
        if not _is_finite(alg):
            raise FloatingPointError("diverged")


def _simulate_central(alg, nodes, config: ExperimentConfig, rounds: int, rng) -> None:
    """
    Short centralized simulation: in each round the nodes contribute new samples (as in
    _simulate) that are pooled; models trained incrementally (warm start, e.g. CentMLP) are
    fitted on all the data received so far in every round, the others once at the end.
    """
    Xs, ys = [], []
    incremental = bool(getattr(getattr(alg, "global_model", None), "warm_start", False))
    for _ in range(rounds):
        for node in nodes:
            n_avail = len(node.X_full)
            if n_avail == 0:
                continue
            n = int(rng.integers(config.node_min_samples, config.node_max_samples + 1))
            idx = rng.choice(n_avail, size=min(n, n_avail), replace=False)
            Xs.append(node.X_full[idx])
            ys.append(node.y_full[idx].reshape(-1, 1))
        if incremental:
            alg.fit(np.vstack(Xs), np.vstack(ys))
    if not incremental:
        alg.fit(np.vstack(Xs), np.vstack(ys))


async def tune_hyperparameters(
    config: ExperimentConfig,
    node_roster: list,
    n_features: int,
    X_val, y_val,
) -> Dict[str, Dict]:
    """Return {alg_name: best_params} for every algorithm with *_grid keys."""
    best_all = {}
    if not config.tune_hyperparameters:
        return best_all  # use the fixed values from the YAML
    rounds = config.tuning_rounds or config.num_rounds

    for name, info in config.algs_config.items():
        grid = _grid_keys(info["params"])
        if not grid:
            continue
        if name in config.tuned_params:
            best_all[name] = config.tuned_params[name]
            continue

        keys = list(grid.keys())
        rows = []
        for values in itertools.product(*grid.values()):
            cand = dict(zip(keys, values))
            alg_cfg = {name: {"class": info["class"], "params": {**info["params"], **cand}}}
            alg = get_algorithm_instance(alg_cfg, n_features, verbose=False, n_clients=len(node_roster))[name]

            rng = np.random.default_rng(config.seed)
            # Large learning rates are expected to diverge: silence NumPy overflow noise
            with np.errstate(all="ignore"):
                try:
                    if name.startswith("Cent"):      # centralized baseline
                        _simulate_central(alg, node_roster, config, rounds, rng)
                    else:
                        await _simulate(alg, node_roster, config, rounds, rng)
                    r2 = float(alg.evaluate(X_val, y_val)["r2"])
                except (FloatingPointError, ValueError, OverflowError):
                    r2 = float("nan")
            if not np.isfinite(r2):
                r2 = -np.inf  # diverged
            rows.append({**cand, "val_r2": r2})
            logger.info(f"[Tuning] {name} {cand} -> val R² = "
                        f"{'diverged' if r2 == -np.inf else f'{r2:.4f}'}")

        df = pd.DataFrame(rows)
        if np.all(df["val_r2"] == -np.inf):
            # Every candidate diverged: fall back to the smallest values and warn
            best_idx = df.sort_values(keys).index[0]
            logger.warning(f"[Tuning] {name}: ALL candidates diverged. Using the smallest "
                           f"values; add lower values to the *_grid keys in the YAML.")
        else:
            best_idx = df["val_r2"].idxmax()
            # Best value on the edge of the grid -> the grid may be too narrow
            for k in keys:
                try:
                    vals = sorted(float(v) for v in df[k].unique())
                except (TypeError, ValueError):    # non-numeric grid (e.g. max_depth "None")
                    continue
                if len(vals) > 1 and float(df.loc[best_idx, k]) in (vals[0], vals[-1]):
                    logger.warning(f"[Tuning] {name}: best {k}={df.loc[best_idx, k]} is on the "
                                   f"edge of {k}_grid {vals}; consider widening it.")
        best = df.loc[best_idx, keys].to_dict()
        best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
        best_all[name] = best
        config.tuned_params[name] = best
        logger.info(f"[Tuning] {name}: best {best} (val R² = {df.loc[best_idx, 'val_r2']:.4f})")

        if config.store_plots:
            out = (RESULTS_DIR / config.dataset_name /
                   f"target_{config.variable_target}" / "tuning")
            out.mkdir(parents=True, exist_ok=True)
            tag = f"{config.node_min_samples}-{config.node_max_samples}-{config.min_samples_train}"
            df.to_csv(out / f"{name.replace(' ', '_')}_{tag}.csv", index=False)

    return best_all
