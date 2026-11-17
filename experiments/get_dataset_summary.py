"""
Summary of the datasets as used in the experiments (after preprocessing and partition).
"""

import numpy as np
import pandas as pd
import common as C

from src.simulation.settings.data import load_raw_data
from src.simulation.data.partitioning import _SPLIT_STATE

def stats(X):
    X = X[:, np.std(X, axis=0) > 0]                      # drop constant columns
    R = np.corrcoef(X, rowvar=False)
    off = np.abs(R[~np.eye(len(R), dtype=bool)])
    vif = np.diag(np.linalg.pinv(R))
    ev = np.linalg.eigvalsh(R)
    return off.mean(), vif.mean(), ev[-1] / max(ev[0], 1e-300)


def main():
    # ---- parameters ------------------------------------------------------
    DATASETS = ["housing", "discharges", "european_survey_adult", "length_of_stay"]
    NUM_NODES = 20                  # as in the experiments
    PARTITION = {"housing": "random"}   # default: natural
    # ----------------------------------------------------------------------
    edir = C.exp_dir("datasets")
    rows = []
    for ds in DATASETS:
        cfg = {**C.base_config(), "num_nodes": NUM_NODES, "partition": PARTITION.get(ds, "natural")}
        node_data, *_ = load_raw_data(ds, C.DATASETS[ds]["target"], cfg)
        
        info = _SPLIT_STATE.get("groups_info")

        X = np.vstack([np.asarray(v[s][0], float) for v in node_data.values()
                       for s in ("train", "val", "test") if s in v and len(v[s][0])])
        rho, vif, cond = stats(X)

        rows.append({"Dataset": C.DATASETS[ds]["label"],
            "Natural groups": info["groups"] if info else np.nan,
            "n (all groups)": info["samples_all_groups"] if info else np.nan,
            "Nodes used": len(node_data),
            "n": len(X), "d": X.shape[1], "Target": C.DATASETS[ds]["target"],
            "mean |rho|": rho, "mean VIF": vif, "cond(R)": cond})
        
        print(rows[-1])
    C.save_table(pd.DataFrame(rows), edir / "table")


if __name__ == "__main__":
    main()
