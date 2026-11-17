"""
Exactness and numerical stability of FedDelta.

(a) Real datasets: the training data of every node is streamed in batches into the
    FedDelta class; after each round the weights are compared with the exact ridge
    solution w* = (A + λD)^{-1} B on the same data.
(b) Synthetic stress test: condition numbers 1e2..1e10 and two λ values,
    rank-deficient updates (batch < d).
Same batch size and number of rounds in (a) and (b). All parameters are set at the
top of main().
"""

import asyncio
import logging

import numpy as np
import pandas as pd

import common as C
from src.algorithms.federated.models.FedDelta import FedDelta, _penalty_matrix
from src.simulation.settings.data import load_raw_data

async def stream(chunks_per_round, d, lam, residual_tol, test=None):
    """
    Feed rounds of (X, y) chunks into FedDelta. Per round: relative weight error w.r.t. the
    exact ridge solution and, if a test set is given, |R²(FedDelta) - R²(exact)|.
    """

    # Initialize FedDelta and the sufficient statistics of the centralized solution
    alg = FedDelta(n_features=d, timeout=1e12, lambda_reg=lam, penalty="constant",
                   residual_tol=residual_tol)
    A = np.zeros((d + 1, d + 1)); B = np.zeros((d + 1, 1))
    D = _penalty_matrix(d + 1)

    # Store round-wise numerical errors and conditioning estimates
    errs, dr2, kap = [], [], []
    if test is not None:
        # Prepare the test set, including the intercept, for R² evaluation
        Xt = np.hstack([np.ones((len(test[0]), 1)), test[0]]); yt = np.asarray(test[1], float).ravel()
        sst = np.sum((yt - yt.mean()) ** 2)
        r2 = lambda w: 1 - np.sum((yt - Xt @ w) ** 2) / sst
    with np.errstate(all="ignore"):
        for chunks in chunks_per_round:
            # Process all node updates belonging to the current round
            for X, y in chunks:
                upd = alg.fit(X, y.reshape(-1, 1))
                await alg.aggregate_update(upd)
                
                # Accumulate the same sufficient statistics used by the centralized ridge solution
                Xi = np.hstack([np.ones((len(X), 1)), X])
                A += Xi.T @ Xi; B += Xi.T @ y.reshape(-1, 1)

            # Finalize the federated round and reset round-level timing metrics
            alg.complete_aggregation_round()
            alg.reset_elapsed_metrics()

            # Compute the centralized reference solution using all data seen so far
            w_ref = np.linalg.solve(A + lam * D, B).ravel()
            w = np.asarray(alg.global_model, dtype=float).ravel()

            # Measure the relative difference between federated and centralized weights
            errs.append(float(
                np.linalg.norm(w - w_ref) / 
                max(np.linalg.norm(w_ref), 1e-300)
            ))

            # Estimate the condition number of the regularized normal-equation matrix
            ev = np.linalg.eigvalsh(A + lam * D)
            kap.append(float(ev[-1] / max(ev[0], 1e-300)))
            if test is not None:
                # Compare test-set performance using the same evaluation data
                dr2.append(abs(r2(w) - r2(w_ref)))
    
    return {
        "err": np.array(errs), 
        "dr2": np.array(dr2) if dr2 else None,
        "refresh": getattr(alg, "n_refreshes", 0), 
        "n": int(round(A[0, 0])),
        "kappa": float(max(kap)), # worst conditioning over the rounds  
    }   


def real_rounds(dataset, batch, max_rounds):
    """
    Build the sequence of streaming rounds for a real dataset.
    """

    # Load the dataset using its natural node partition
    cfg = {
        "num_nodes": 20, 
        "scale_target": True, 
        "partition": "natural"
    }

    node_data, _, _, d, _ = load_raw_data(dataset, C.DATASETS[dataset]["target"], cfg)

    # Collect each node's training data
    trains = [v["train"] for v in node_data.values()]

    # Combine node-level test sets into a single evaluation set
    test = (np.vstack([v["test"][0] for v in node_data.values()]),
            np.concatenate([np.ravel(v["test"][1]) for v in node_data.values()]))

    # Build rounds by taking consecutive batches from each node
    rounds, r = [], 0
    while r < max_rounds:
        chunks = [(X[r * batch:(r + 1) * batch], y[r * batch:(r + 1) * batch])
                  for X, y in trains if len(X) > r * batch]
        chunks = [(X, y) for X, y in chunks if len(X) > 0]
        
        # Stop when no node has any training samples left
        if not chunks:
            break
        
        rounds.append(chunks)
        r += 1
    
    return rounds, d, test


def synthetic_rounds(d, kappa, n_rounds, batch, seed=0):
    """
    Generate synthetic streaming data with controlled feature conditioning.
    """
    rng = np.random.default_rng(seed)

    # Generate an orthogonal feature-space transformation
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))

    # Set feature scales to target the condition number of XᵀX
    s = np.geomspace(1.0, 1.0 / np.sqrt(kappa), d)        # cond(XᵀX) ≈ kappa
    w = rng.normal(size=d)
    out = []
    for _ in range(n_rounds):

        # Generate a batch with the prescribed feature scales and correlation structure
        X = rng.normal(size=(batch, d)) * s @ Q.T

        # Generate targets from a linear model with additive noise
        y = X @ w + rng.normal(0, 0.1, batch)
        
        out.append([(X, y)])
    
    return out

def main():
    # ---- parameters ------------------------------------------------------
    DATASETS = [
        "housing", 
        "discharges", 
        "european_survey_adult", 
        "length_of_stay"
    ]

    # Shared batch size and maximum number of rounds
    BATCH = 20                   
    N_ROUNDS = 50                  
    
    # Dimensionality of the synthetic stress-test data
    D_SYN = 50                      

    # Target condition numbers for the synthetic feature matrix
    KAPPAS = [1e2, 1e4, 1e6, 1e8, 1e10]   

    # Compare weak regularization with the value specified in the YAML configuration
    LAM = float(np.atleast_1d(C.base_config()["algorithms"]["FedDelta"]["lambda_reg"])[0])
    LAMBDAS = [1e-6, LAM]      

    # Residual threshold controlling FedDelta's numerical safeguard
    RESIDUAL_TOL = 1e-8         
    
    # Ensure that each synthetic batch has fewer samples than features
    assert BATCH < D_SYN, "BATCH must be < D_SYN so that synthetic updates are rank-deficient"

    # Suppress refactorization logs; the number of refactorizations is recorded separately
    logging.getLogger("FedDelta").setLevel(logging.ERROR) 
    logging.getLogger(
        "src.algorithms.federated.models.FedDelta"
    ).setLevel(logging.ERROR)

    # Create the output directory and obtain machine precision
    edir = C.exp_dir("exactness")
    eps = np.finfo(float).eps

    # ----------------------------------------------------------------------
    # (a) Real-dataset exactness evaluation
    # ----------------------------------------------------------------------
    print("Exactness (a) real datasets")
    real_rows = []
    
    for ds in DATASETS:

        # Construct the streaming rounds and evaluate FedDelta against the centralized solution
        rounds, d, test = real_rounds(ds, BATCH, N_ROUNDS)
        res = asyncio.run(stream(rounds, d, LAM, RESIDUAL_TOL, test))

        # Collect numerical accuracy, conditioning, and safeguard statistics
        real_rows.append({"Dataset": C.DATASETS[ds]["label"], "d": d, "n": res["n"], "Rounds": len(res["err"]),
                          "kappa(P)": res["kappa"], "eps*kappa": eps * res["kappa"],
                          "Max rel. weight error": res["err"].max(), "Final rel. weight error": res["err"][-1],
                          "Max |dR2|": res["dr2"].max(), "Refactorizations": res["refresh"]})
        
        print(f"   {ds}: d={d}, rounds={len(res['err'])}, kappa={res['kappa']:.1e}, "
              f"max err={res['err'].max():.1e}, max |dR2|={res['dr2'].max():.1e}")

    # ----------------------------------------------------------------------
    # (b) Synthetic numerical stress test
    # ----------------------------------------------------------------------
    print("Exactness (b) synthetic stress test")
    syn_rows = []

    # Evaluate every combination of regularization strength and target conditioning
    for lm in LAMBDAS:
        for k in KAPPAS:

            # Generate synthetic rounds and compare FedDelta with the centralized reference
            res = asyncio.run(stream(synthetic_rounds(D_SYN, k, N_ROUNDS, BATCH), D_SYN, lm, RESIDUAL_TOL))

            # Record the worst and final numerical errors for each configuration
            syn_rows.append({"lambda": lm, "kappa(XtX) target": k, "d": D_SYN, "Rounds": N_ROUNDS,
                             "kappa(P)": res["kappa"], "eps*kappa": eps * res["kappa"],
                             "Max rel. weight error": res["err"].max(),
                             "Final rel. weight error": res["err"][-1], "Refactorizations": res["refresh"]})
            print(f"   κ={k:.0e} λ={lm:g}: max κ(P)={res['kappa']:.1e}, max err={res['err'].max():.1e}")

    # Save the real-data and synthetic results in separate tables
    C.save_table(pd.DataFrame(real_rows), edir / "table_real")
    C.save_table(pd.DataFrame(syn_rows), edir / "table_synthetic")

if __name__ == "__main__":
    main()
