import numpy as np
import logging

from sklearn.model_selection import train_test_split
from src.simulation.data.utils import *

from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

_SPLIT_SEED = 42                       # node partition: fixed for every run
# Seed of the train/val/test split (changed per run when resplit_per_run is on).
# `last_partition` keeps the node partition so it can be re-split without re-reading the data.
# `global_split`: if True, the train/val/test split is drawn over the whole pool of samples,
# independently of the nodes, so every partition has exactly the same train and test samples.
_SPLIT_STATE = {"seed": 42, "last_partition": None, "global_split": False}

MIN_NODE_SIZE = 10                     # minimum number of samples of a node


def set_split_seed(seed: int) -> None:
    _SPLIT_STATE["seed"] = int(seed)

def set_global_split(flag: bool) -> None:
    _SPLIT_STATE["global_split"] = bool(flag)

# Partition across nodes (set by the loader from the config)
_PARTITION = {"mode": "natural", "alpha": 0.5}

def set_partition(mode: str = "natural", alpha: float = 0.5) -> None:
    _PARTITION["mode"] = mode or "natural"
    _PARTITION["alpha"] = float(alpha)


# Every partition function returns {node: (X, y, ids)}, where ids are the positions of the
# node's samples in the pool. The ids let the global split be the same for all partitions.

def split_into_nodes(X, y, n_nodes):
    """IID split: random permutation (fixed seed) cut into n_nodes equal parts."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    indices = np.random.default_rng(_SPLIT_SEED).permutation(len(X))
    return {f"client_{i}": (X[idx], y[idx], idx)
            for i, idx in enumerate(np.array_split(indices, n_nodes))}

def split_dirichlet(X, y, n_nodes, alpha, min_size=MIN_NODE_SIZE, max_tries=1000):
    """
    Label-skew non-IID partition: 10 target-quantile bins (deciles), each spread over the
    nodes with proportions ~ Dir(alpha). Small alpha = strong heterogeneity.
    The proportions are redrawn until every node has at least `min_size` samples, so no
    sample is discarded (Dirichlet partition conditioned on a minimum node size).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(_SPLIT_SEED)
    bins = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
    labels = np.digitize(y, bins)
    for attempt in range(1, max_tries + 1):
        parts = [[] for _ in range(n_nodes)]
        for b in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == b))
            props = rng.dirichlet(np.full(n_nodes, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    else:
        raise RuntimeError(f"Dirichlet(alpha={alpha}): no draw with all {n_nodes} nodes "
                           f">= {min_size} samples in {max_tries} attempts")
    logger.info(f"[Partition] Dirichlet(alpha={alpha}) accepted after {attempt} draw(s)")
    out = {}
    for i, p in enumerate(parts):
        idx = np.array(p, dtype=int)
        out[f"client_{i}"] = (X[idx], y[idx], idx)
    return out

def partition_nodes(X, y, n_nodes, groups=None):
    """
    Partition according to the configured mode:
      natural   -> one node per group (hospital, CCAA, facility); random if no groups
      random    -> IID uniform split
      dirichlet -> label-skew non-IID split, Dir(alpha), minimum node size
    For datasets with natural groups, random/dirichlet re-distribute exactly the same
    samples as the natural partition into the same number of nodes, so the partitions
    differ only in how the data is assigned to nodes.
    """
    mode = _PARTITION["mode"]

    # Natural groups available in the dataset (before keeping the n_nodes largest ones)
    if groups is not None:
        g_counts = np.unique(np.asarray(groups), return_counts=True)[1]
        _SPLIT_STATE["groups_info"] = {"groups": len(g_counts),
                                       "samples_all_groups": int(g_counts.sum())}
    else:
        _SPLIT_STATE["groups_info"] = None

    if mode == "iid":
        # IID split of all samples into n_nodes, ignoring any natural grouping (scalability in K)
        node_data = split_into_nodes(X, y, n_nodes)
    elif groups is not None:
        natural = split_by_group(X, y, groups, n_nodes)
        if mode == "natural":
            node_data = natural
        else:
            # same pool (and same pool order) as the natural partition, only re-distributed
            X = np.vstack([v[0] for v in natural.values()])
            y = np.concatenate([v[1] for v in natural.values()])
            n_nodes = len(natural)
            node_data = (split_dirichlet(X, y, n_nodes, _PARTITION["alpha"]) if mode == "dirichlet"
                         else split_into_nodes(X, y, n_nodes))
    else:
        node_data = (split_dirichlet(X, y, n_nodes, _PARTITION["alpha"]) if mode == "dirichlet"
                     else split_into_nodes(X, y, n_nodes))
    sizes = [len(v[1]) for v in node_data.values()]
    logger.info(f"[Partition] mode={mode} | nodes={len(sizes)} | "
                f"size min/median/max = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
    return node_data

def split_by_group(X, y, groups, n_nodes=None):
    """Natural partition: one node per group value (largest groups first).
    The pool is the concatenation of the kept groups in this order; ids index that pool."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)
    values, counts = np.unique(groups, return_counts=True)
    keep = values[np.argsort(-counts)]
    if n_nodes is not None:
        keep = keep[:n_nodes]
    # Nodes need >= MIN_NODE_SIZE samples to be split into train/val/test
    keep = [g for g in keep if (groups == g).sum() >= MIN_NODE_SIZE]
    out, offset = {}, 0
    for g in keep:
        m = groups == g
        ids = np.arange(offset, offset + m.sum())
        out[f"node_{g}"] = (X[m], y[m], ids)
        offset += m.sum()
    return out

def _global_labels(n_pool, test_ratio, val_ratio, seed):
    """0 = train, 1 = val, 2 = test for every sample of the pool (independent of the nodes)."""
    # Own seed stream: must not coincide with the permutation of the IID partition (seed 42)
    perm = np.random.default_rng([int(seed), 0x5EED]).permutation(n_pool)
    n_test = int(round(test_ratio * n_pool))
    n_val = int(round(val_ratio * n_pool))
    lab = np.zeros(n_pool, dtype=int)
    lab[perm[:n_test]] = 2
    lab[perm[n_test:n_test + n_val]] = 1
    return lab

def build_tabular_client_data(
    node_data: dict,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    shuffle: bool = True,
    random_state: int = None,
) -> tuple:
    """
    Shared post-processing for all regression loaders.

    - Splits the data into train / val / test:
        global_split=False: per node (70/15/15 of each node).
        global_split=True:  over the whole pool, independently of the partition, so all
                            partitions have exactly the same train, val and test samples.
    - Fits a MinMaxScaler on concatenated training splits only (no leakage).
    - Returns {node_id: {"train": (X, y), "val": (X, y), "test": (X, y)}}.
    """
    _SPLIT_STATE["last_partition"] = node_data
    if random_state is None:
        random_state = _SPLIT_STATE["seed"]

    # ── Step 1: train / val / test split ─────────────────────────────
    splits = {}
    if _SPLIT_STATE["global_split"]:
        n_pool = 1 + max(int(v[2].max()) for v in node_data.values() if len(v[2]))
        lab = _global_labels(n_pool, test_ratio, val_ratio, random_state)
        rng = np.random.default_rng([int(random_state), 0xC0DE])
        for node_id, (X, y, ids) in node_data.items():
            # Shuffle the node's samples (as train_test_split does in the per-node split):
            # the Dirichlet partition builds each node decile by decile, so without this
            # the node would deliver its data sorted by the target over the rounds.
            order = rng.permutation(len(ids))
            X = np.asarray(X, dtype=float)[order]
            y = np.asarray(y, dtype=float).ravel()[order]
            l = lab[np.asarray(ids)[order]]
            splits[node_id] = (X[l == 0], X[l == 1], X[l == 2], y[l == 0], y[l == 1], y[l == 2])
    else:
        for node_id, (X, y, _ids) in node_data.items():
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).ravel()
            # First cut: train vs (val + test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size = val_ratio + test_ratio,
                shuffle = shuffle,
                random_state = random_state,
            )
            # Second cut: val vs test (relative size within the temp set)
            relative_test = test_ratio / (val_ratio + test_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size = relative_test,
                shuffle = shuffle,
                random_state = random_state,
            )
            splits[node_id] = (X_train, X_val, X_test, y_train, y_val, y_test)

    # ── Step 2: fit scaler on training data only ──────────────────────
    X_train_all = np.vstack([v[0] for v in splits.values()])
    scaler = MinMaxScaler().fit(X_train_all)

    # ── Step 3: apply scaler and package ─────────────────────────────
    tr = lambda X: scaler.transform(X) if len(X) else X   # a small node may have an empty split
    data = {}
    for node_id, (X_train, X_val, X_test, y_train, y_val, y_test) in splits.items():
        data[node_id] = {
            "train": (tr(X_train), y_train),
            "val":   (tr(X_val),   y_val),
            "test":  (tr(X_test),  y_test),
        }

    return data, len(data)