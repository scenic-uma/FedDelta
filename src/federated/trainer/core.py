import logging
import numpy as np
import time
import inspect

from typing import List, Tuple, Dict, Optional, Any
from src.federated.nodes.federated import FederatedNode
from src.algorithms.federated.base import FederatedAlgorithm
from src.federated.coordination.leader_election import choose_leader, demote_leader

from .round import run_federated_round
from src.federated.coordination.gossip_protocol import promote_to_leader
import random as _random


def elect_without_gossip(ready_nodes, strategy: str):
    """Leader election baselines for the ablation (no gossip messages)."""
    if not ready_nodes:
        return None
    if strategy == "random":
        leader = _random.choice(ready_nodes)
    else:  # "fixed": deterministic first available node (behaves like a fixed server)
        leader = min(ready_nodes, key=lambda n: str(n.id))
    promote_to_leader(leader)
    for n in ready_nodes:
        n.leader = leader
        n.leaderNotification = True
        n.is_leader = n is leader
    return leader
from .evaluation import evaluation_nodes, evaluate_on_nodes
from .utils import (log_round_header, update_history,
    log_metrics, split_algorithms, init_history,
    log_node_status, collect_diagnostics, check_participant_mismatch,
    determine_ready_nodes)
from .node_failure import handle_node_lifecycle

logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------
def run_centralized_algorithms(
    centralized_algs: List[Tuple[str, FederatedAlgorithm]],
    X_full: np.ndarray,
    y_full: np.ndarray,
    X_round: np.ndarray,
    y_round: np.ndarray,
    X_val_global: Optional[np.ndarray],
    y_val_global: Optional[np.ndarray],
    eval_nodes: List,
    history: Dict,
    mse_scale: float = 1.0
) -> None:
    """
    Fits each centralized baseline on the data accumulated from participating nodes,
    then evaluates it on the test splits of the SAME nodes used for the federated
    algorithms this round (no pooled test set).
    """
    for name, alg in centralized_algs:
        start = time.perf_counter()

        if "X_val" in inspect.signature(alg.fit).parameters:
            # Online fit: this round's data plus validation set
            alg.fit(X_round, y_round, X_val_global, y_val_global)
        else:
            # Batch fit: retrain on the entire accumulated dataset
            alg.fit(X_full, y_full)

        # Training time only (evaluation excluded, as for federated algorithms)
        fit_time = time.perf_counter() - start

        global_metrics = evaluate_on_nodes(alg, eval_nodes)
        global_metrics["exec_time"] = fit_time
        update_history(history, name, global_metrics, mse_scale=mse_scale)

def final_evaluation(algorithms: Dict, node_roster: List, history: Dict, mse_scale: float) -> None:
    """
    Final model of every algorithm evaluated on ALL nodes (fixed population).
    Stored in history[alg]["final"]; used by summary tables and statistical tests.
    """
    for name, alg in algorithms.items():
        if name not in history or alg.global_model is None:
            continue
        try:
            m = evaluate_on_nodes(alg, node_roster)
        except Exception as exc:  # model never trained (e.g. no successful round)
            logger.warning(f"[Final evaluation] {name} skipped: {exc}")
            continue
        m["mse"] *= mse_scale
        history[name]["final"] = m

async def run_federated_algorithms(
    federated_algs: List[Tuple[str, FederatedAlgorithm]],
    ready_nodes: List[FederatedNode],
    node_roster: List[FederatedNode],
    current_leader: FederatedNode,
    eval_nodes: List[FederatedNode],
    history: Dict,
    dataset_name: str,
    effective_round: int,
    verbose: bool,
    mse_scale: float = 1.0,
    leader_crash: bool = False,
) -> Tuple[bool, List, Dict]:
    """
    Runs one federated round for every algorithm in federated_algs.
 
    A round is considered successful if at least one algorithm reaches quorum
    and produces a global model update. Algorithms that fail quorum are skipped
    but do not abort the others.
    """
    # Nothing to do if there are no federated algorithms
    if not federated_algs:
        return False, {}
 
    round_succeeded = False
    alg_participants: Dict[str, List] = {}  # tracks participant sets per algorithm
 
    for name, alg in federated_algs:
        # Reset participants  and time for each algorithm's round
        current_leader.reset_round()
        alg.reset_elapsed_metrics()

        # Execute one training round: local fits → aggregation → global update
        success, global_metrics, node_metrics = \
            await run_federated_round(
                name          = name,
                alg           = alg,
                ready_nodes   = ready_nodes,
                leader        = current_leader,
                node_roster   = node_roster,
                eval_nodes    = eval_nodes,
                verbose       = verbose,
                leader_crash  = leader_crash,
            )
 
        # Record participant list for consistency check across algorithms
        alg_participants[name] = current_leader.participants.copy()
 
        if not success:
            # Quorum not reached — log and move on to the next algorithm
            logger.info("-" * 50)
            logger.info(
                f"Round {effective_round + 1} | {name}: not enough updates "
                f"(minimum quorum not reached)."
            )
            logger.info("-" * 50)
            continue
 
        # Mark the round as successful and store results
        round_succeeded = True
        update_history(history, name, global_metrics, node_metrics, mse_scale=mse_scale)

    # Warn if different algorithms trained on different node subsets
    round_participants = check_participant_mismatch(alg_participants)

    # Reset participants for each algorithm's round
    current_leader.reset_round()
 
    return round_succeeded, round_participants
#--------------------------------------------------------------------------------
async def run_training(
    algorithms: List[Tuple[str, FederatedAlgorithm]],
    node_roster: List[FederatedNode],
    num_rounds: int,
    min_samples_train: int,
    node_min_samples: int,
    node_max_samples: int,
    dataset_name: str,
    max_skipped_rounds: int,
    node_failure_rate: float = 0.0,
    failure_interval: int = None,
    metrics: List[FederatedNode] = ["mse", "r2", "exec_time"],
    X_test_global: Optional[np.ndarray] = None,
    y_test_global: Optional[np.ndarray] = None,
    X_val_global: Optional[np.ndarray] = None,
    y_val_global: Optional[np.ndarray] = None,
    gossip_fanout: int = 3,
    gossip_rounds: int = 3,
    target_scale: float = 1.0,
    leader_failure_rate: float = 0.0,
    leader_downtime: int = 2,
    samples_mode: str = "fixed",
    stream_jitter: float = 0.2,
    stream_min_batch: int = 1,
    election: str = "gossip",
    reference: str = "FedDelta",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], int, int]:
    mse_scale = target_scale ** 2

    # Separate algorithms into federated and centralized groups
    federated_algs, centralized_algs = split_algorithms(algorithms)

    # Centralized baselines train on exactly the batches committed by this algorithm (C5)
    ref_alg = algorithms.get(reference) or (federated_algs[0][1] if federated_algs else None)

    # Build the history dict with empty lists for each metric
    history = init_history(algorithms, metrics)
    history["faults"] = []
    dup_prev = 0   # rejected_duplicates is cumulative per algorithm: log the per-round increase

    # Stream mode: each node receives ~len(train)/num_rounds samples per round
    if samples_mode == "stream":
        for n in node_roster:
            n.stream_batch = int(np.ceil(len(n.X_full) / max(1, num_rounds)))
            n.stream_jitter = stream_jitter
            if stream_jitter <= 0:
                # Deterministic schedule: k equal batches (sizes differ by at most one) of at
                # least stream_min_batch samples, sent in the LAST k rounds, so that every node
                # delivers all its data and all nodes finish together in round num_rounds
                # (the quorum can always be reached and no tiny last batch is sent)
                n_tr = len(n.X_full)
                k = max(1, min(num_rounds, n_tr // max(1, stream_min_batch)))
                sizes = [len(c) for c in np.array_split(np.arange(n_tr), k)]
                n.stream_sizes = [0] * (num_rounds - len(sizes)) + sizes
        min_samples_train = 1

    # State variables shared across rounds
    current_leader: Optional[FederatedNode] = None
    X_full = y_full = None               # accumulated training dataset
    total_samples: int = 0               # grows as new node data is added
    skipped_rounds: int = 0              # consecutive rounds with no updates
    effective_round: int = 0             # rounds that produced at least one update
    absolute_round: int = 0 

    # Loop over each training round
    while effective_round < num_rounds:
        absolute_round += 1

        # Log round header for monitoring
        if verbose:
            log_round_header(effective_round, num_rounds)

        # Demote the previous leader so a fresh election can take place
        if current_leader:
            demote_leader(current_leader)

        # Optionally drop nodes permanently on a specific round

        if node_failure_rate > 0:
            handle_node_lifecycle(
                node_roster       = node_roster, 
                absolute_round    = absolute_round, 
                node_failure_rate = node_failure_rate, 
                failure_interval  = failure_interval, 
                verbose           = verbose
            )
        
        # Crash-recovery: nodes (e.g. crashed leaders) come back after their downtime (C4)
        for n in node_roster:
            if n.recover_at is not None and absolute_round >= n.recover_at:
                n.reconnect()
                n.recover_at = None

        # Nodes that are up evaluate this round (same set for EVERY algorithm, Cent included)
        eval_nodes = evaluation_nodes(node_roster)

        # Determine which nodes are ready to participate in this round
        ready_nodes = determine_ready_nodes(
            node_roster, node_min_samples,
            node_max_samples, min_samples_train, verbose
        )

        if verbose and node_failure_rate > 0:
            logger.info("-" * 50)  
            log_node_status(node_roster, effective_round)

        # Elect a coordinator among the ready nodes using gossip-based leader election
        if election == "gossip":
            current_leader = await choose_leader(
                ready_nodes = ready_nodes, 
                fanout      = gossip_fanout, 
                rounds      = gossip_rounds,
                verbose     = verbose
            )
        else:
            # Ablation baselines: random leader, or always the first available node (server-like)
            current_leader = elect_without_gossip(ready_nodes, election)

        # If no leader could be elected, log a warning and stop training
        if not current_leader:
            logger.warning(f"Round {effective_round + 1}: Leader election failed. "
                      f"Skipping this round and trying again next one.")
            skipped_rounds += 1

            if skipped_rounds >= max_skipped_rounds:
                logger.error("Maximum consecutive skipped rounds reached due to repeated leader election failures.")
                break

            continue

        if "node_status" not in history:
            history["node_status"] = {n.id: [] for n in node_roster}
            history["effective_rounds"] = []

        for n in node_roster:
            if n.interrupted:                          # highest priority: dropped
                status = 0
            elif not n.active:                         # exhausted: no data left, not interrupted
                status = 3
            elif n in ready_nodes:                     # active, ready and with enough data
                status = 2
            elif n.active and not n.ready_to_train:    # active but not ready yet
                status = 1
            else:                                      # active and ready, but not enough data
                status = 1
            
            history["node_status"][n.id].append(status)

        # Leader crash (C6): the elected leader fails after collecting the updates and
        # before consolidating them; the whole round is aborted for every algorithm.
        leader_crash = leader_failure_rate > 0 and np.random.rand() < leader_failure_rate

        # ── Federated round ───────────────────────────────────────────
        # Run one update step for every federated algorithm and collect results
        round_succeeded, round_participants = await run_federated_algorithms(
            federated_algs  = federated_algs,
            ready_nodes     = ready_nodes,
            node_roster     = node_roster,
            dataset_name    = dataset_name,
            current_leader  = current_leader,
            eval_nodes      = eval_nodes,
            history         = history,
            effective_round = effective_round,
            verbose         = verbose,
            mse_scale       = mse_scale,
            leader_crash    = leader_crash,
        )

        # Fault log (one row per absolute round)
        dup_now = sum(a.rejected_duplicates for _, a in federated_algs)
        history["faults"].append({
            "round": absolute_round,
            "n_down": sum(n.interrupted for n in node_roster),
            "n_ready": len(ready_nodes),
            "leader": current_leader.id,
            "leader_crash": bool(leader_crash),
            "round_committed": bool(round_succeeded),
            # batches re-sent after a failed round / missed deadline / crash (not lost)
            "resent_batches": sum(n.resent_last.pop(ref_alg, 0) for n in node_roster),
            "rejected_duplicates": dup_now - dup_prev,   # rejected in this round
        })
        dup_prev = dup_now

        if leader_crash:
            logger.warning(f"[Fault] Leader {current_leader.id} crashed: round aborted, "
                           f"down for {leader_downtime} round(s).")
            current_leader.interrupt()
            current_leader.recover_at = absolute_round + leader_downtime
            current_leader = None

        if round_succeeded:
            effective_round += 1
            skipped_rounds = 0  # reset the skip counter on a successful round
            history["effective_rounds"].append(absolute_round)
 
            # Centralized data = batches committed this round by the reference algorithm (C5)
            parts = [n.committed_data(ref_alg) for n in node_roster]
            parts = [(X, y) for X, y in parts if X is not None]
            if parts:
                X_round = np.vstack([X for X, _ in parts])
                y_round = np.concatenate([y.ravel() for _, y in parts])
                X_full = X_round if X_full is None else np.vstack([X_full, X_round])
                y_full = y_round if y_full is None else np.concatenate([y_full, y_round])
            else:
                X_round, y_round = np.empty((0, 0)), np.empty((0,))

            # Track cumulative sample count across all rounds
            total_samples = X_full.shape[0] if X_full is not None else 0
            history["Total samples"].append(total_samples)

            # Consistency check: FedDelta must contain exactly the samples Cent has
            fd = algorithms.get("FedDelta")
            if fd is not None and ref_alg is fd and X_full is not None:
                n_fd = int(round(fd.global_terms["A"][0, 0]))
                if n_fd != total_samples:
                    logger.warning(f"[Consistency] FedDelta has {n_fd} samples, Cent has {total_samples}.")

            # ── Centralized round ─────────────────────────────────────
            # Re-fit centralized baselines on the updated dataset
            if X_full is not None:
                run_centralized_algorithms(
                    centralized_algs   = centralized_algs,
                    X_full             = X_full,
                    y_full             = y_full,
                    X_round            = X_round,
                    y_round            = y_round,
                    X_val_global       = X_val_global,
                    y_val_global       = y_val_global,
                    eval_nodes         = eval_nodes,
                    history            = history,
                    mse_scale          = mse_scale,
                )
 
            # Print a summary of this round's metrics to the log
            log_metrics(history, effective_round, absolute_round, len(round_participants))

        else:
            # No algorithm produced an update this round
            skipped_rounds += 1
            logger.info("-" * 50)
            logger.info(f"Round {effective_round + 1} skipped | No federated algorithm executed.")
            logger.info("-" * 50)

        # Abort if too many consecutive rounds have been skipped
        if skipped_rounds >= max_skipped_rounds:
            logger.info("Maximum consecutive skipped rounds reached. Stopping training.")
            break

    # ── Post-training diagnostics ─────────────────────────────────────────────
    # Collect subspace alignment and node drift metrics from algorithms that support them
    collect_diagnostics(federated_algs, history)

    # Data conservation (fault tolerance): checked from the node ledgers, independently of
    # the protocol and of the centralized baselines
    history["integrity"] = data_integrity(algorithms.get("FedDelta"), node_roster)

    # Final models evaluated on ALL nodes (fixed population) -> tables and statistics
    final_evaluation(algorithms, node_roster, history, mse_scale)

    return history


def data_integrity(fd, node_roster) -> Dict[str, float]:
    """
    Data conservation check for FedDelta at the end of a run.

    From the batch ledger of every node (not from the protocol):
      - batches collected, committed (incorporated) and still pending (not yet sent or
        re-sent); lost = collected - committed - pending must be 0;
      - batches committed more than once (exactly-once, C1): must be 0;
      - the aggregated Gram matrix A of FedDelta is compared with sum(X^T X) recomputed
        from the raw data of the committed batches, each counted once; A[0,0] is the
        number of samples incorporated (intercept column of ones).
    """
    if fd is None or not hasattr(fd, "global_terms"):
        return {}
    d = fd.n_features + 1
    A_ref = np.zeros((d, d))
    n_ref = collected = committed = pending = twice = 0
    for n in node_roster:
        done = n.committed.get(fd, set())
        collected += len(n.batches)
        committed += len(done)
        pending += len(n.pending_ids(fd))
        twice += n.double_commits.get(fd, 0)
        for b in done:
            Xb = fd.add_intercept(np.asarray(n.batches[b][0], dtype=float))
            A_ref += Xb.T @ Xb
            n_ref += Xb.shape[0]
    A_fd = fd.global_terms["A"]
    rel = float(np.linalg.norm(A_fd - A_ref) / max(np.linalg.norm(A_ref), 1e-300))
    out = {
        "batches_collected": collected,
        "batches_committed": committed,
        "batches_pending":   pending,
        "batches_lost":      collected - committed - pending,
        "batches_committed_twice": twice,
        "duplicates_rejected_by_leader": int(fd.rejected_duplicates),
        "samples_expected":  n_ref,
        "samples_in_model":  float(A_fd[0, 0]),
        "A_rel_error":       rel,
    }
    if out["batches_lost"] or twice or abs(out["samples_in_model"] - n_ref) > 0.5:
        logger.warning(f"[Integrity] data conservation violated: {out}")
    else:
        logger.info(f"[Integrity] OK: {committed} batches ({n_ref} samples) incorporated once, "
                    f"{pending} pending, rel. error of A = {rel:.1e}")
    return out

