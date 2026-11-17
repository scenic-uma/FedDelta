"""
Fault tolerance of FedDelta under the protocol.

(a) Client dropout: in each round, with equal probability, either a fraction p of the
    active nodes fails or a fraction p of the failed nodes recovers (at least two nodes
    always remain active).
(b) Leader crash: the elected leader crashes with probability q after collecting the
    updates; the round is aborted and the contributions are retained and re-sent.
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

import common as C


def fault_stats(hist):
    f = [x for h in hist for x in h.get("faults", [])]
    n_runs = max(1, len(hist))
    n_rounds = len(f)                                   # absolute rounds over all runs
    aborted = sum(not x.get("round_committed", True) for x in f)
    resent = sum(x.get("resent_batches", 0) for x in f)
    integ = [h.get("integrity", {}) for h in hist]
    get = lambda k: [x[k] for x in integ if k in x]
    return {"Rounds/run": n_rounds / n_runs,
            "Committed rounds/run": (n_rounds - aborted) / n_runs,
            "Nodes down/round": float(np.mean([x.get("n_down", 0) for x in f])) if f else 0.0,
            "Aborted rounds (%)": 100 * aborted / max(1, n_rounds),
            "Leader crashes/run": sum(bool(x.get("leader_crash")) for x in f) / n_runs,
            "Resent batches/aborted round": resent / max(1, aborted),
            "Duplicates rejected": sum(x.get("rejected_duplicates", 0) for x in f),
            # data conservation, summed / worst case over runs
            "Batches committed/run": float(np.mean(get("batches_committed"))) if integ else np.nan,
            "Batches pending at end/run": float(np.mean(get("batches_pending"))) if integ else np.nan,
            "Batches lost": int(np.sum(get("batches_lost"))) if integ else -1,
            "Batches committed twice": int(np.sum(get("batches_committed_twice"))) if integ else -1,
            "Max |samples in model - expected|": float(np.max(np.abs(np.subtract(
                get("samples_in_model"), get("samples_expected"))))) if integ else np.nan,
            "Max rel. error of A": float(np.max(get("A_rel_error"))) if integ else np.nan}


def main():
    # ---- parameters ------------------------------------------------------
    DATASETS = ["housing", "discharges", "european_survey_adult", "length_of_stay"]
    NODES = {"housing": 20, "discharges": 20, "european_survey_adult": 19, "length_of_stay": 5}
    ALGS = [C.FEDDELTA, C.CENT]     # Cent only as internal check
    DROPOUT = [0.0, 0.1, 0.3, 0.5]  # fraction of nodes that fail / recover per round
    LEADER = [0.0, 0.1, 0.2, 0.3]   # probability that the leader crashes in a round
    QUORUM = 0.4                    # quorum as fraction of the nodes
    PARTITION = "natural"           # natural partition (Housing: random)
    N_RUNS = 1                     # independent runs per (dataset, failure level)
    N_ROUNDS = 30                   # committed rounds per run (fewer if the data run out)
    SAMPLES_MIN = 50                # new samples per node and round (min)
    SAMPLES_MAX = 50                # new samples per node and round (max)
    TUNE = "yaml"                   # baseline tuning: "yaml" | "on" | "off"
    FORCE = False                   # True: recompute even if cached
    # ----------------------------------------------------------------------

    run_args = SimpleNamespace(runs=N_RUNS, rounds=N_ROUNDS, tune=TUNE, quick=False)
    samples = [{"node_min_samples": SAMPLES_MIN, "node_max_samples": SAMPLES_MAX,
                "min_samples_train": 1}]
    edir = C.exp_dir("fault_tolerance")

    rows = []
    for ds in DATASETS:
        m = math.ceil(QUORUM * NODES[ds])
        base = {}
        for kind, levels in (("dropout", DROPOUT), ("leader", LEADER)):
            for lv in levels:
                print(f"fault_tolerance {ds}: {kind} = {lv} (quorum {m})")
                over = dict(partition=PARTITION, samples_mode="fixed", node_samples_list=samples,
                            robustness_runs=N_RUNS, max_skipped_rounds=1000, quorum=m)
                if kind == "dropout":
                    over.update(node_failure_rate=lv, failure_interval=1)
                else:
                    over.update(leader_failure_rate=lv)
                cfg = C.make_config(run_args, [ds], ALGS, **over)
                tag = C.tag(run_args, ds, kind, lv, f"b{SAMPLES_MIN}-{SAMPLES_MAX}", f"m{m}", "integ")
                hist = C.histories_of(C.run(cfg, tag, edir, FORCE), ds)
                fd, ce = C.final_values(hist, C.FEDDELTA, "r2"), C.final_values(hist, C.CENT, "r2")
                if lv == 0:                 # level 0 (no failures) is the reference
                    base[kind] = np.nanmean(fd)
                    # both no-failure runs are the same simulation: check it explicitly
                    if len(base) == 2 and not np.isclose(base["dropout"], base["leader"], atol=1e-12):
                        print(f"  WARNING {ds}: no-failure references differ "
                              f"({base['dropout']:.12f} vs {base['leader']:.12f})")
                rows.append({"Dataset": C.DATASETS[ds]["label"], "Nodes": hist[0].get("n_nodes"),
                             "Quorum": m, "Failure": kind, "Level": lv,
                             "R2 mean": np.nanmean(fd),
                             "R2 sd": np.nanstd(fd, ddof=1) if len(fd) > 1 else 0,
                             "Change vs no failures": np.nanmean(fd) - base[kind],
                             "|FedDelta-Cent| max": np.nanmax(np.abs(fd - ce)),
                             **fault_stats(hist)})
    C.save_table(pd.DataFrame(rows), edir / "table")


if __name__ == "__main__":
    main()
