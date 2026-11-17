from typing import Dict, List, Optional
import numpy as np
 
 
class DiagnosticsCollector:
    """
    Collects and stores per-round diagnostic data for the FedDelta algorithm.
    Decouples diagnostic logic from the main training loop.
    """
 
    def __init__(self, max_history: Optional[int] = None):
        """
        Args:
            max_history: Maximum number of rounds to keep in memory.
                         If None, history grows unbounded.
        """

        if isinstance(max_history, str):
            self.max_history = None
        else:
            self.max_history = max_history
 
        # Per-round update matrices (U = delta_A + mu*I)
        self.update_matrices_history: List[np.ndarray] = []
        # Singular value spectra of each round's update matrix
        self.update_spectra_history: List[np.ndarray] = []
        # Snapshot of the global matrix P before each round's update
        self.global_matrix_history: List[np.ndarray] = []
        # Number of nodes that participated in each round
        self.nodes_per_round_history: List[int] = []
        # Per-node local A_i matrices, grouped by round
        self.node_update_history: List[List[np.ndarray]] = []
 
        # Accumulators for the current in-progress round
        self._current_round_updates: List[np.ndarray] = []
        self._current_round_nodes: int = 0
 
    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------
 
    def record_node_update(self, A_i: np.ndarray) -> None:
        """
        Called by aggregate_update once per node contribution.
        Stores A_i for drift analysis at the end of the round.
        """
        # Copy only if the caller may mutate A_i after this call
        self._current_round_updates.append(A_i.copy())
        self._current_round_nodes += 1
 
    def record_round_update(
        self,
        U: np.ndarray,
        P_prev: Optional[np.ndarray],
        effective_rank_fn,
    ) -> None:
        """
        Called once per round inside complete_aggregation_round, after U is built
        but before the global matrix is updated.
 
        Args:
            U:                 Round update matrix (delta_A + mu*I).
            P_prev:            Global matrix P from the previous round (or None).
            effective_rank_fn: Callable that takes a matrix and returns its erank.
        """
        _, S, _ = np.linalg.svd(U, full_matrices=False)
 
        self._append(self.update_spectra_history, S)
        self._append(self.update_matrices_history, U.copy())
        self.nodes_per_round_history.append(self._current_round_nodes)
 
        if P_prev is not None:
            self._append(self.global_matrix_history, P_prev.copy())
 
    def finalize_round(self) -> None:
        """
        Closes the current round: moves per-node accumulators to history
        and resets them for the next round.
        """
        self._append(self.node_update_history, self._current_round_updates)
        self._current_round_updates = []
        self._current_round_nodes = 0
 
    # ------------------------------------------------------------------
    # On-demand metrics (computed lazily, not stored redundantly)
    # ------------------------------------------------------------------
 
    @property
    def last_erank_round(self) -> Optional[float]:
        """Effective rank of the most recent round's update matrix."""
        if not self.update_matrices_history:
            return None
        return self._effective_rank(self.update_matrices_history[-1])
 
    @property
    def last_erank_global(self) -> Optional[float]:
        """Effective rank of the most recent global matrix snapshot."""
        if not self.global_matrix_history:
            return None
        return self._effective_rank(self.global_matrix_history[-1])
 
    # ------------------------------------------------------------------
    # Analysis methods (previously on FedDelta)
    # ------------------------------------------------------------------
 
    def analyze_node_drift(self) -> List[Dict]:
        """
        Computes per-node drift for all rounds with respect to
        the previous global matrix.
        """
        results = []
 
        for round_idx, node_updates in enumerate(self.node_update_history):
            if round_idx >= len(self.global_matrix_history):
                continue
 
            P_prev = self.global_matrix_history[round_idx]
            nodes_round = len(node_updates)
 
            for node_id, A_i in enumerate(node_updates):
                drift_value = self._batch_drift_score(A_i, P_prev)
                results.append({
                    "round": round_idx + 1,
                    "node_id": node_id,
                    "nodes_participating": nodes_round,
                    "drift": float(drift_value),
                    "drift_per_node": float(drift_value / nodes_round) if nodes_round > 0 else None,
                })
 
        results.sort(key=lambda x: (x["round"], -x["drift"]))
        return results
 
    def analyze_subspace_alignment(self, energy: float = 0.9, eps: float = 1e-12) -> List[Dict]:
        """
        Computes subspace alignment and spectral diagnostics between
        consecutive round update matrices.
        """
        matrices = self.update_matrices_history
        nodes_history = self.nodes_per_round_history
 
        if len(matrices) < 2:
            return []
 
        results = []
        for i in range(1, len(matrices)):
            M_prev, M_curr = matrices[i - 1], matrices[i]
 
            U_prev, S_prev = self._dominant_subspace(M_prev, energy)
            U_curr, S_curr = self._dominant_subspace(M_curr, energy)
 
            # Subspace alignment (dimension-matched)
            k = min(U_prev.shape[1], U_curr.shape[1])
            sigma = np.linalg.svd(U_prev[:, :k].T @ U_curr[:, :k], compute_uv=False)
            alignment = float(np.mean(sigma))
 
            # Spectral drift (padded to equal length)
            p_prev = self._spectrum_distribution(S_prev)
            p_curr = self._spectrum_distribution(S_curr)
            max_len = max(len(p_prev), len(p_curr))
            spectral_drift = float(np.linalg.norm(
                np.pad(p_prev, (0, max_len - len(p_prev))) -
                np.pad(p_curr, (0, max_len - len(p_curr)))
            ))
 
            erank_prev = self._effective_rank(M_prev)
            erank_curr = self._effective_rank(M_curr)
 
            results.append({
                "round": i + 1,
                "nodes_participating": nodes_history[i] if i < len(nodes_history) else None,
                "alignment": round(alignment, 4),
                "spectral_drift": round(spectral_drift, 6),
                "erank_prev": round(erank_prev, 3),
                "erank_curr": round(erank_curr, 3),
                "erank_diff": round(abs(erank_curr - erank_prev), 4),
                "cond_prev": round(self._condition_number(S_prev, eps), 2),
                "cond_curr": round(self._condition_number(S_curr, eps), 2),
            })
 
        return results
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _append(self, store: list, item) -> None:
        """Appends to a list, evicting the oldest entry if max_history is set."""
        store.append(item)
        if self.max_history is not None:
            if len(store) > self.max_history:
                store.pop(0)
 
    @staticmethod
    def _effective_rank(matrix: np.ndarray) -> float:
        """Effective rank via singular value entropy."""
        sv = np.linalg.svd(matrix, compute_uv=False)
        sv_sum = np.sum(sv)
        if sv_sum == 0:
            return 0.0
        p = sv / sv_sum
        p = p[p > 0]
        return float(np.exp(-np.sum(p * np.log(p))))
 
    @staticmethod
    def _batch_drift_score(A_i: np.ndarray, A_global: np.ndarray, eps: float = 1e-12) -> float:
        """Frobenius norm between trace-normalized local and global matrices."""
        return np.linalg.norm(
            A_i / (np.trace(A_i) + eps) - A_global / (np.trace(A_global) + eps),
            ord="fro"
        )
 
    @staticmethod
    def _dominant_subspace(M: np.ndarray, energy: float):
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        if np.sum(S) == 0:
            return U[:, :1], S
        k = np.searchsorted(np.cumsum(S) / np.sum(S), energy) + 1
        return U[:, :k], S
 
    @staticmethod
    def _spectrum_distribution(S: np.ndarray) -> np.ndarray:
        total = np.sum(S)
        return S / total if total > 0 else S
 
    @staticmethod
    def _condition_number(S: np.ndarray, eps: float) -> float:
        if len(S) == 0:
            return np.inf
        S_nz = S[S > eps * np.max(S)]
        if len(S_nz) == 0:
            return np.inf
        return float(np.max(S_nz) / np.min(S_nz))
 