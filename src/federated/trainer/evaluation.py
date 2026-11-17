"""
Federated evaluation (no pooled test set is ever built).

Each node evaluates the model on its OWN test split and shares only
sufficient statistics: n, SSE, Σy, Σy² and its local R².
From these the exact pooled metrics are rebuilt:
    MSE = Σ SSE / N
    R²  = 1 - Σ SSE / SST,    SST = Σ y² - (Σ y)² / N
plus fairness across nodes (each node weighs the same):
    r2_macro (mean), r2_median, r2_worst (minimum)

The same functions are used for federated and centralized models, so all
algorithms are evaluated on exactly the same nodes and the same samples.
"""
from typing import Dict, List

import numpy as np


def evaluation_nodes(node_roster: List) -> List:
    """Nodes that can evaluate this round: every node that is up (not interrupted).
    Nodes with no training data left are up and still have a test split."""
    return [n for n in node_roster if not n.interrupted]


def node_stats(alg, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Sufficient statistics of one node's local evaluation."""
    y = np.asarray(y_test, dtype=float).ravel()
    n = len(y)
    if n == 0:   # a small node may have no test samples under the global split
        return {"n": 0.0, "sse": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "r2_local": float("nan")}
    m = alg.evaluate(X_test, y)
    return {
        "n":        float(n),
        "sse":      float(m["mse"]) * n,
        "sum_y":    float(y.sum()),
        "sum_y2":   float((y ** 2).sum()),
        "r2_local": float(m["r2"]),
    }


def pooled_metrics(stats: List[Dict[str, float]]) -> Dict[str, float]:
    """Exact pooled MSE/R² + fairness metrics from per-node statistics."""
    stats = [s for s in stats if s["n"] > 0]
    if not stats:
        return {}
    N = sum(s["n"] for s in stats)
    sse = sum(s["sse"] for s in stats)
    sum_y = sum(s["sum_y"] for s in stats)
    sst = sum(s["sum_y2"] for s in stats) - sum_y ** 2 / N
    local = np.array([s["r2_local"] for s in stats], dtype=float)
    local = local[np.isfinite(local)]
    return {
        "mse":       sse / N,
        "r2":        1.0 - sse / sst if sst > 0 else float("nan"),
        "r2_macro":  float(local.mean()) if local.size else float("nan"),
        "r2_median": float(np.median(local)) if local.size else float("nan"),
        "r2_worst":  float(local.min()) if local.size else float("nan"),
        "n_eval_nodes": len(stats),
    }


def evaluate_on_nodes(alg, nodes: List) -> Dict[str, float]:
    """Evaluate any model (federated or centralized) on the given nodes' test splits."""
    return pooled_metrics([node_stats(alg, *n.data["test"]) for n in nodes])
