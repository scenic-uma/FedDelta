"""
Statistical comparison of algorithms across independent runs.

For each metric the final-round value of every run is used (runs are paired:
same seed, same data stream). Outputs:
  - descriptive: mean, std, median, IQR, 95% CI
  - omnibus:     Friedman test + mean ranks (paired, non-parametric)
  - pairwise:    reference vs others -> Wilcoxon signed-rank and Welch t-test,
                 Holm-corrected p-values, Vargha-Delaney A12, Hedges' g,
                 mean difference with 95% CI
  - equivalence: paired TOST (reference vs exact baselines) with margin delta
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Metrics where lower is better
LOWER_IS_BETTER = {"mse", "exec_time", "w_rel_error"}
SKIP_KEYS = {"__gossip__", "node_status", "effective_rounds", "election",
             "Total samples", "lambda_reg", "run_seed", "faults", "integrity", "n_features", "n_nodes"}


# -----------------------------------------------------------------------------
# Data extraction
# -----------------------------------------------------------------------------
def final_round_table(histories: List[Dict], metric: str) -> pd.DataFrame:
    """
    Rows = runs, columns = algorithms.
    Uses the final evaluation on ALL nodes (history[alg]["final"]) when available,
    otherwise the metric at the last common round (e.g. exec_time).
    """
    algs = [k for k, v in histories[0].items()
            if k not in SKIP_KEYS and isinstance(v, dict) and "global" in v]
    data = {}
    for alg in algs:
        finals = [h.get(alg, {}).get("final", {}).get(metric) for h in histories]
        if all(f is not None for f in finals):
            data[alg] = [float(f) for f in finals]
            continue
        series = [h.get(alg, {}).get("global", {}).get(metric, []) for h in histories]
        if not all(series):
            continue
        last = min(len(s) for s in series) - 1
        data[alg] = [float(s[last]) for s in series]
    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# Effect sizes and helpers
# -----------------------------------------------------------------------------
def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) + 0.5 P(X = Y). 0.5 = no effect; >0.71 or <0.29 = large."""
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return 0.0 if np.mean(x) == np.mean(y) else np.inf * np.sign(np.mean(x) - np.mean(y))
    j = 1 - 3 / (4 * (nx + ny) - 9)  # small-sample correction
    return float(j * (np.mean(x) - np.mean(y)) / sp)


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def mean_ci(x: np.ndarray, level: float = 0.95):
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return float(np.mean(x)), float(np.mean(x))
    h = stats.t.ppf((1 + level) / 2, len(x) - 1) * stats.sem(x)
    return float(np.mean(x) - h), float(np.mean(x) + h)


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


def safe_welch(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 and np.std(y) == 0:
        return 1.0 if np.mean(x) == np.mean(y) else 0.0
    return float(stats.ttest_ind(x, y, equal_var=False).pvalue)


def paired_tost(x: np.ndarray, y: np.ndarray, delta: float) -> float:
    """Two one-sided paired t-tests; small p => equivalent within ±delta."""
    d = x - y
    n = len(d)
    if n < 2:
        return float("nan")
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0 if abs(np.mean(d)) < delta else 1.0
    se = sd / np.sqrt(n)
    p_low = 1 - stats.t.cdf((np.mean(d) + delta) / se, n - 1)
    p_up = stats.t.cdf((np.mean(d) - delta) / se, n - 1)
    return float(max(p_low, p_up))


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def run_statistical_analysis(
    histories: List[Dict],
    out_dir: Path,
    metrics: List[str],
    reference: str = "FedDelta",
    equivalent_to: Optional[List[str]] = None,
    equivalence_margin: Optional[Dict[str, float]] = None,
    alpha: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Compute all tests, save CSV return the DataFrames."""
    n_runs = len(histories)
    if n_runs < 2:
        logger.info("[Stats] Need at least 2 runs, skipping.")
        return {}
    if n_runs < 10:
        logger.warning(f"[Stats] Only {n_runs} runs: low power "
                       f"(Wilcoxon cannot reach p<0.05 with fewer than 6). Use >= 10.")

    equivalent_to = equivalent_to or ["Cent"]
    margins = {"r2": 0.01, **(equivalence_margin or {})}
    out_dir.mkdir(parents=True, exist_ok=True)

    desc_rows, omni_rows, pair_rows, eq_rows = [], [], [], []

    for metric in metrics:
        df = final_round_table(histories, metric)
        if df.empty:
            continue

        # Descriptive
        for alg in df.columns:
            x = df[alg].to_numpy()
            lo, hi = mean_ci(x)
            q1, q3 = np.percentile(x, [25, 75])
            desc_rows.append({"metric": metric, "algorithm": alg, "n": len(x),
                              "mean": x.mean(), "std": x.std(ddof=1) if len(x) > 1 else 0.0,
                              "median": np.median(x), "q1": q1, "q3": q3,
                              "ci95_low": lo, "ci95_high": hi})

        # Omnibus Friedman (needs >= 3 algorithms)
        if df.shape[1] >= 3:
            try:
                chi2, p = stats.friedmanchisquare(*[df[c] for c in df.columns])
            except ValueError:
                chi2, p = np.nan, np.nan
            sign = 1 if metric in LOWER_IS_BETTER else -1
            ranks = (sign * df).rank(axis=1).mean()
            omni_rows.append({"metric": metric, "test": "Friedman", "statistic": chi2, "p_value": p,
                              **{f"rank_{a}": r for a, r in ranks.items()}})

        # Pairwise: reference vs others
        if reference not in df.columns:
            continue
        ref = df[reference].to_numpy()
        others = [c for c in df.columns if c != reference]
        rows = []
        for alg in others:
            x = df[alg].to_numpy()
            d = ref - x
            lo, hi = mean_ci(d)
            rows.append({"metric": metric, "reference": reference, "algorithm": alg,
                         "mean_diff": d.mean(), "diff_ci95_low": lo, "diff_ci95_high": hi,
                         "p_wilcoxon": safe_wilcoxon(ref, x), "p_welch": safe_welch(ref, x),
                         "A12": vargha_delaney_a12(ref, x), "hedges_g": hedges_g(ref, x)})
        if rows:
            for key in ("p_wilcoxon", "p_welch"):
                adj = holm([r[key] if np.isfinite(r[key]) else 1.0 for r in rows])
                for r, a in zip(rows, adj):
                    r[f"{key}_holm"] = a
            for r in rows:
                r["significant"] = r["p_wilcoxon_holm"] < alpha
            pair_rows.extend(rows)

        # Equivalence (TOST) vs exact baselines
        if metric in margins:
            for alg in equivalent_to:
                if alg in df.columns:
                    p = paired_tost(ref, df[alg].to_numpy(), margins[metric])
                    eq_rows.append({"metric": metric, "reference": reference, "algorithm": alg,
                                    "margin": margins[metric], "p_tost": p,
                                    "equivalent": bool(p < alpha)})

    results = {
        "descriptive": pd.DataFrame(desc_rows),
        "omnibus": pd.DataFrame(omni_rows),
        "pairwise": pd.DataFrame(pair_rows),
        "equivalence": pd.DataFrame(eq_rows),
    }
    for name, df in results.items():
        if not df.empty:
            df.to_csv(out_dir / f"stats_{name}.csv", index=False)

    logger.info(f"[Stats] Saved statistical analysis to {out_dir}")
    _log_summary(results)
    return results


def _log_summary(results: Dict[str, pd.DataFrame]) -> None:
    pair, eq = results["pairwise"], results["equivalence"]
    if not pair.empty:
        cols = ["metric", "algorithm", "mean_diff", "p_wilcoxon_holm", "p_welch_holm", "A12"]
        logger.info("[Stats] Pairwise vs reference:\n%s", pair[cols].to_string(index=False))
    if not eq.empty:
        logger.info("[Stats] Equivalence (TOST):\n%s", eq.to_string(index=False))
