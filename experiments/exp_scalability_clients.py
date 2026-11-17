"""
Scalability in the number of clients K.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import common as C

def per_round(metric, hist, a):
    """
    Return the mean and run-level standard deviation of a round metric.
    """
    arr = C.curves(hist, a, metric)
    return (np.nanmean(arr), np.nanstd(np.nanmean(arr, axis=1), ddof=1)) if arr.size else (np.nan, np.nan)

def main():

    # SPARCS split at random (IID) into K nodes, ignoring the hospitals, so K can vary freely
    DATASET = "discharges"

    # Algorithms included in the scalability comparison
    ALGS = [
        C.FEDDELTA, 
        C.CENT, 
        "FedGD (FedAvg)", 
        "FedGD (FedProx)", 
        "FedSCAFFOLD",
        "FedGD (FedNova)", 
        "FedMLP (FedAvg)", 
        "FedGD (FedAvgM)"
    ]

    # Number of clients evaluated
    CLIENTS = [10, 20, 50, 100, 200]

    # Number of independent runs and communication rounds per configuration
    N_RUNS = 10           
    N_ROUNDS = 10               

    # Number of new samples assigned to each client
    SAMPLES_MIN = 50           
    SAMPLES_MAX = 110              

    # Hyperparameter tuning configuration
    TUNE = "yaml"                   # baseline tuning: "yaml" | "on" | "off"

    # Output and execution options
    SAVE_FIGURES = False
    FORCE = False # Recompute results even if cached results are available
    
    # ----------------------------------------------------------------------

    # Build the arguments expected by the shared experiment utilities
    run_args = SimpleNamespace(
        runs=N_RUNS, 
        rounds=N_ROUNDS, 
        tune=TUNE, 
        quick=False
    )

    # Define the range of samples assigned to each client
    samples = [{
        "node_min_samples": SAMPLES_MIN, 
        "node_max_samples": SAMPLES_MAX,
        "min_samples_train": 1
    }]

    # Load the base configuration and retrieve the gossip protocol parameters
    base = C.base_config()
    fan, grounds = base.get("gossip_fanout", 3), base.get("gossip_rounds", 3)

    # Create the output directory
    edir = C.exp_dir("scalability_clients")

    rows = []

    # Evaluate each configuration with a different number of clients
    for k in CLIENTS:
        print(f"scalability_clients {DATASET} K={k}")

        # Configure an IID partition with a fixed number of samples per client
        cfg = C.make_config(run_args, [DATASET], ALGS, num_nodes=k, partition="iid",
                        samples_mode="fixed", node_samples_list=samples)
        
        # Generate a unique cache identifier for this configuration
        tag = C.tag(run_args, DATASET, "K", k, f"b{SAMPLES_MIN}-{SAMPLES_MAX}")
        
        # Execute the experiment or load its cached results
        hist = C.histories_of(C.run(cfg, tag, edir, FORCE), DATASET)

        # Estimate the average number of gossip election messages per round
        msgs = np.mean([np.mean([f["n_ready"] * min(fan, max(f["n_ready"] - 1, 0)) * grounds + f["n_ready"]
                                 for f in h["faults"]]) for h in hist])

        # Extract timing and performance metrics for each algorithm
        for a in ALGS:

            # Compute aggregation, local fitting, and total round-time statistics
            agg, fit, rnd = per_round("agg_time", hist, a), per_round("fit_time", hist, a), per_round("exec_time", hist, a)

            # Extract final test-set R² values across independent runs
            r2 = C.final_values(hist, a, "r2")

            # Store timing, communication, election, and predictive performance metrics
            rows.append({
                "Dataset": C.DATASETS[DATASET]["label"], "K": hist[0].get("n_nodes", k),
                "Algorithm": C.short(a),
                "Leader agg. time (ms)": 1e3 * agg[0], "Leader agg. time sd (ms)": 1e3 * agg[1],
                "Client fit time (ms)": 1e3 * fit[0], "Client fit time sd (ms)": 1e3 * fit[1],
                "Round time (ms)": 1e3 * rnd[0], "Round time sd (ms)": 1e3 * rnd[1],
                "KB per round": per_round("comm_bytes", hist, a)[0] / 1e3,
                "Election msgs/round": msgs,
                "Final R2": np.nanmean(r2), "Final R2 sd": np.nanstd(r2, ddof=1)
            })

    # Save the collected scalability results
    df = pd.DataFrame(rows)
    C.save_table(df, edir / "table")

    # Optionally generate scalability plots
    if SAVE_FIGURES:

        # Select the metrics to display and whether to use a logarithmic y-axis
        panels = [("Leader agg. time (ms)", True), ("Round time (ms)", True),
                  ("KB per round", True), ("Final R2", False)]
        fig, axes = C.plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.4), squeeze=False)

        # Plot each metric against the number of clients
        for ax, (m, logy) in zip(axes[0], panels):
            for a in ALGS:

                # Skip centralized metrics that are not meaningful for these panels
                if a == C.CENT and m in ("Leader agg. time (ms)", "KB per round"):
                    continue
                s = df[df.Algorithm == C.short(a)].sort_values("K")
                ax.plot(s.K, s[m], **C.style(a))
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Clients K"); ax.set_title(m)

        # Add a shared legend and save the figure
        C.legend_below(fig, axes, ALGS); fig.subplots_adjust(bottom=0.32)
        C.save_fig(fig, edir / "fig_scalability")


if __name__ == "__main__":
    main()
