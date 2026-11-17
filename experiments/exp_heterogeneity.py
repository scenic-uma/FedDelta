"""
Statistical heterogeneity across nodes.

The same data of each dataset is split into the same number of nodes in four ways:
  random        IID split (nodes are statistically identical)
  natural       one node per hospital / region / facility (not available for Housing)
  dirichlet-1   label skew: each decile of y spread over the nodes with Dir(1.0)
  dirichlet-0.1 strong label skew, Dir(0.1)

"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

import common as C


# Maps each partition name to the corresponding partitioning strategy
# and, when applicable, its Dirichlet concentration parameter.
PARTS = {"random": ("random", None), "natural": ("natural", None),
         "dirichlet-1": ("dirichlet", 1.0), "dirichlet-0.1": ("dirichlet", 0.1)}

# Datasets for which no natural grouping variable is available.
NO_GROUPS = {"housing"}         


def main():
    
    # Datasets included in the heterogeneity evaluation.
    DATASETS = [
        "housing", 
        "discharges", 
        "european_survey_adult", 
        "length_of_stay"
    ]

    # Federated algorithms and centralized reference method.
    ALGS = [
        C.FEDDELTA, 
        C.CENT, 
        "FedGD (FedAvg)", 
        "FedGD (FedProx)", 
        "FedSCAFFOLD",
        "FedMLP (FedAvg)", 
        "FedGD (FedAvgM)"
    ]
    
    N_RUNS = 1             # Number of independent runs per dataset and partition.
    N_ROUNDS = 30           # Number of federated communication rounds per run.

    SAMPLES_MIN = 50        # Minimum number of new samples per node and round
                            # when using fixed-size batches.
    SAMPLES_MAX = 110       # Maximum number of new samples per node and round
                            # when using fixed-size batches.

    GLOBAL_SPLIT = True     # If True, all partitioning strategies share the
                            # same global train/test split.

    SAMPLES_MODE = "stream" # "stream": each node distributes its entire training
                            # dataset across N_ROUNDS rounds.
                            # "fixed": each node contributes between
                            # SAMPLES_MIN and SAMPLES_MAX samples per round.

    STREAM_JITTER = 0.0     # Controls variability in streaming batch sizes.
                            # Zero produces deterministic, equally sized batches,
                            # ensuring that all training samples are used.

    STREAM_MIN_BATCH = 10   # Minimum number of samples per streaming batch.
                            # Nodes with fewer remaining samples send them all
                            # together rather than creating smaller batches.

    TUNE = "on"             # Baseline hyperparameter tuning:
                            # "yaml": use the values specified in the YAML file.
                            # "on": enable hyperparameter tuning.
                            # "off": disable hyperparameter tuning.

    # Regularization parameters used by centralized and federated ridge methods.
    LAMBDA = {
        "housing": 10,
        "discharges": 1e-4,
        "european_survey_adult": 1,
        "length_of_stay": 1e-4
    }

    SAVE_FIGURES = True     # Generate bar plots showing final mean R² and
                            # worst-node R² for each partitioning strategy.

    FORCE = False           # If True, recompute all experiments even when
                            # cached results are available.

    # ----------------------------------------------------------------------

    # Build the common experiment arguments shared across all configurations.
    run_args = SimpleNamespace(runs=N_RUNS, rounds=N_ROUNDS, tune=TUNE, quick=False)

    # Configure the per-node sample limits used by the data partitioning process.
    samples = [{"node_min_samples": SAMPLES_MIN, "node_max_samples": SAMPLES_MAX,
                "min_samples_train": 1}]

    # Directory where experiment outputs and result tables will be stored.
    edir = C.exp_dir("heterogeneity")

    # Accumulate the results from all datasets, partitions, and algorithms.
    rows = []

    # ---- Run experiments -------------------------------------------------
    for ds in DATASETS:
        for pname, (mode, alpha) in PARTS.items():

            # Skip natural partitioning when the dataset has no grouping variable.
            if ds in NO_GROUPS and mode == "natural":
                continue
            print(f"heterogeneity {ds} {pname}")

            # Define the data partitioning and streaming configuration.
            over = {
                "partition": mode, 
                "samples_mode": SAMPLES_MODE, 
                "node_samples_list": samples,
                "global_split": GLOBAL_SPLIT, 
                "stream_jitter": STREAM_JITTER,
                "stream_min_batch": STREAM_MIN_BATCH
            }

            # Specify the Dirichlet concentration parameter for label-skewed splits.
            if alpha is not None:
                over["dirichlet_alpha"] = alpha

            # Build the experiment configuration for the current dataset
            # and partitioning strategy.
            cfg = C.make_config(
                run_args, 
                [ds], 
                ALGS, 
                lambda_scope="ridge", 
                **over
            )

            # Apply the dataset-specific ridge regularization parameter to FedDelta.
            cfg["algorithms"][C.FEDDELTA]["lambda_reg"] = [LAMBDA[ds]]

            # Construct a unique experiment identifier based on the streaming
            # configuration and whether a global train/test split is shared.
            smp = (f"stream{STREAM_JITTER:g}-min{STREAM_MIN_BATCH}" if SAMPLES_MODE == "stream"
                   else f"b{SAMPLES_MIN}-{SAMPLES_MAX}")
            
            tag = C.tag(run_args, ds, pname, smp, "gs" if GLOBAL_SPLIT else "ns")

            # Execute the experiment or load cached results when available.
            hist = C.histories_of(C.run(cfg, tag, edir, FORCE), ds)

            # Retrieve centralized ridge regression performance as a reference.
            cent = C.final_values(hist, C.CENT, "r2")

            # Extract and aggregate the final evaluation metrics for each algorithm.
            for a in ALGS:

                # Mean test R² across independent runs.
                v, w = C.final_values(hist, a, "r2"), C.final_values(hist, a, "r2_worst")
                
                rows.append({
                    "Dataset": C.DATASETS[ds]["label"], 
                    "Partition": pname,
                    "Algorithm": C.short(a), 
                    "Nodes": hist[0].get("n_nodes"),

                    # Average predictive performance across independent runs. 
                    "R2 mean": np.nanmean(v), 

                    # Standard deviation of predictive performance across runs.
                    "R2 sd": np.nanstd(v, ddof=1) if len(v) > 1 else 0,

                    # Difference between centralized ridge R² and the
                    # corresponding federated algorithm's R².
                    "Gap to Cent": np.nanmean(cent - v),

                    # Average worst-node performance across independent runs.
                    "Worst-node R2": np.nanmean(w)}
                )

    # ---- Save aggregated results ----------------------------------------

    # Convert the accumulated experiment results into a DataFrame.
    df = pd.DataFrame(rows)

    # Export the complete results table.
    C.save_table(df, edir / "table")

    # ---- Generate figures -----------------------------------------------

    if SAVE_FIGURES:

        # Partitioning strategies considered in the experiment.
        parts = list(PARTS)

        # Generate separate figures for average and worst-node performance.
        for metric, name, yl in (("R2 mean", "r2", r"Final test $R^2$"),
                                 ("Worst-node R2", "worst", r"Worst-node $R^2$")):
            n = len(DATASETS)
            fig, axes = C.plt.subplots(1, n, figsize=(2.8 * n, 2.5), squeeze=False)
            width = 0.8 / len(ALGS)
            for ax, ds in zip(axes[0], DATASETS):
                sub = df[df.Dataset == C.DATASETS[ds]["label"]]
                ps = [p for p in parts if p in set(sub.Partition)]
                for j, a in enumerate(ALGS):
                    vals = [sub[(sub.Partition == p) & (sub.Algorithm == C.short(a))][metric].mean() for p in ps]
                    ax.bar(np.arange(len(ps)) + (j - len(ALGS) / 2 + 0.5) * width, np.maximum(vals, -0.2),
                           width, color=C.COLORS[a], label=C.short(a))
                ax.set_xticks(range(len(ps)), ps, rotation=30, ha="right")
                ax.set_title(C.DATASETS[ds]["label"]); ax.set_ylim(bottom=max(-0.2, ax.get_ylim()[0]))
            axes[0][0].set_ylabel(yl)

            # Reserve space for the legend and rotated x-axis labels.
            C.legend_below(fig, axes, ALGS, bars=True); fig.subplots_adjust(bottom=0.38)

            # Save the figure in the experiment output directory.
            C.save_fig(fig, edir / f"fig_{name}")


if __name__ == "__main__":
    main()
