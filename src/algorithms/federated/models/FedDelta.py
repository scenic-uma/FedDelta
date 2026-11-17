from typing import Dict, List, Optional
from scipy.linalg import eigh
from ..base import FederatedAlgorithm
from ...diagnostics_collector import DiagnosticsCollector
from scipy.linalg import pinvh

import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

def _penalty_matrix(dim: int) -> np.ndarray:
    """Ridge penalty on the coefficients only (intercept, column 0, is not penalized),
    as in sklearn Ridge / standard ridge regression."""
    D = np.eye(dim)
    D[0, 0] = 0.0
    return D


class FedDelta(FederatedAlgorithm):
    """
    Federated Delta strategy for linear regression.
    """

    def __init__(
        self,
        n_features: int,
        timeout: float,
        lambda_reg: float = 0.0, 
        penalty: str = None,
        numeric_checks: bool = False,
        erank_diagnostics: bool = False,
        erank_diagnostics_max_history: Optional[int] = None,
        residual_tol: float = 1e-8,
        verbose: bool = False
    ):
        """
        Initialize the FedDelta algorithm.
        """
        
        self.n_features = n_features

        self.timeout = timeout
        
        initial_weights = np.zeros(self.n_features)  # +1 for bias term
        
        super().__init__(
            initial_weights = initial_weights,
            lambda_reg      = lambda_reg,
            penalty         = penalty,
            verbose         = verbose
        )

        # Whether to perform numerical checks on matrices
        self.numeric_checks = numeric_checks 
        # Previous regularization parameter lambda
        self.prev_lambda_reg = 0.0    

        self.erank_diagnostics = erank_diagnostics

        self.residual_tol = residual_tol
        self.n_refreshes = 0

        # Accumulates global statistics
        self.global_terms = {
            'A': np.zeros((self.n_features + 1, self.n_features + 1)),  # Sum of local XᵀX matrices
            'B': np.zeros((self.n_features + 1, 1)),         # Sum of local Xᵀy vectors
            'P': None,        # P = A_global + lambda*I (regularizado acumulado)
            'P_inv': None
        }

        # These matrices temporarily accumulate local contributions from all nodes in a single round
        self.round_terms = {
            'A': np.zeros((self.n_features + 1, self.n_features + 1)),  # Sum of local XᵀX matrices
            'B': np.zeros((self.n_features + 1, 1)),             # Sum of local Xᵀy vectors
        }

        # Diagnostics collector — None when diagnostics are disabled
        self.diagnostics: Optional[DiagnosticsCollector] = (
            DiagnosticsCollector(max_history = erank_diagnostics_max_history)
            if erank_diagnostics
            else None
        )

    def flops_info(self, n: int) -> dict:
        """
        Returns a breakdown of theoretical FLOPs for one local training round.
        """
        d_eff = self.n_features + 1  # intercept included

        gram_cost     = n * d_eff**2  # X^T X  —  O(n * d^2)
        crossp_cost   = n * d_eff     # X^T y  —  O(n * d)
        eigh_cost     = d_eff**3      # eigendecomposition via eigh  —  O(d^3)
        woodbury_cost = d_eff**3      # Sherman-Morrison-Woodbury update  —  O(d^3)

        return {
            "gram":       gram_cost,
            "cross_prod": crossp_cost,
            "eigh":       eigh_cost,
            "woodbury":   woodbury_cost,
            "total_flops": float(gram_cost + crossp_cost + eigh_cost + woodbury_cost)
        }
    
    def analyze_node_drift(self) -> List[Dict]:
        """
        Delegates to DiagnosticsCollector. Returns [] if diagnostics are off.
        """
        if self.diagnostics is None:
            return []
        return self.diagnostics.analyze_node_drift()
    
    def analyze_subspace_alignment(self, energy: float = 0.9, eps: float = 1e-12) -> List[Dict]:
        """
        Delegates to DiagnosticsCollector. Returns [] if diagnostics are off.
        """
        if self.diagnostics is None:
            return []
        return self.diagnostics.analyze_subspace_alignment(energy=energy, eps=eps)
    
    def is_positive_semidefinite(self, matrix: np.ndarray, tol: float = 1e-12)-> bool:
        """
        Check if a matrix is positive semidefinite (PSD) within a numerical tolerance.
        """
        # Compute eigenvalues (for symmetric matrix use eigh)
        eigvals = np.linalg.eigh(matrix)[0]
        # Check if all eigenvalues are >= -tol (allowing small negatives due to floating point errors)
        return np.all(eigvals >= -tol)
    
    def _run_numeric_checks(self, U: np.ndarray, S: np.ndarray, Q: np.ndarray) -> None:
        """Runs optional numeric sanity checks on U and its eigendecomposition."""
        if not np.allclose(U, U.T, atol=1e-10):
            logger.warning(f"[{self.__class__.__name__}] delta_A + mu*I is not symmetric. Decomposition may be unreliable.")
 
        if not self.is_positive_semidefinite(U):
            if self.verbose:
                logger.warning(f"[{self.__class__.__name__}] delta_A is not positive semidefinite. Results may be unstable.")
        elif self.verbose:
            logger.info(f"[{self.__class__.__name__}] delta_A is positive semidefinite, proceeding with update.")
 
        recon_error = np.linalg.norm(U - Q @ np.diag(S) @ Q.T)
        if self.verbose:
            logger.info(f"[{self.__class__.__name__}] Eigendecomposition reconstruction error: {recon_error:.2e}")
    
    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------
    def fit(
        self, X_i: np.ndarray, 
        y_i: np.ndarray, 
        features: List = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Local computation at the node:
        A_i = Xᵀ X
        B_i = Xᵀ y
        Returns the tuple (A_i, B_i).
        """

        # Add intercept term (bias)
        X_i = self.add_intercept(X_i)

        if features:
            features_with_intercept = [0] + [f + 1 for f in features]
            X_i = X_i[:, features_with_intercept]

        start = time.perf_counter()
        A_i = X_i.T.dot(X_i) # Compute local Gram matrix A_i
        B_i = X_i.T.dot(y_i) # Compute local cross-product B_i
        elapsed = time.perf_counter() - start
        
        return (A_i, B_i, X_i.shape[0], elapsed)

    async def aggregate_update(
        self, 
        update: tuple[np.ndarray, np.ndarray],
        features: List = None
    ) -> None:
        """
        Aggregates contributions from a single node into the global model state.
        A = Σ A_i, B = Σ B_i
        """

        async with self.lock:
            A_i, B_i, n_i, elapsed = update
            # Track each node's fit duration for round time calculation
            self.round_fit_times.append(elapsed)
            
            start = time.perf_counter()

            if features:
                # Map features to global indices: add 0 for intercept
                global_idx = [0] + [f + 1 for f in features]
                idx_matrix = np.ix_(global_idx, global_idx)

                #Partial update: node has a subset of features (clustered)
                self.round_terms['A'][idx_matrix] += A_i
                self.round_terms['B'][global_idx] += B_i.reshape(-1, 1)
            else:
                # Accumulate local A_i and B_i into global matrices
                self.round_terms['A'] += A_i
                self.round_terms['B'] += B_i.reshape(-1, 1)

            self.round_elapsed += time.perf_counter() - start

            if self.diagnostics is not None:
                self.diagnostics.record_node_update(A_i)
    
    def comm_floats(self):
        """
        Uplink: symmetric A_i (upper triangle) + b_i. Downlink: the same global
        statistics, broadcast so that any node can act as the next leader.
        """
        d = self.n_features + 1
        stats = d * (d + 1) // 2 + d
        return stats, stats

    def discard_round(self) -> None:
        """Drop the partial sums of an incomplete round (A and B are cumulative)."""
        self.round_terms['A'].fill(0)
        self.round_terms['B'].fill(0)
        super().discard_round()

    def update_global_matrices(self):
        """
        Updates the global aggregated matrices A_global and B_global 
        using the current round's local contributions.
        """
        self.global_terms['A'] += self.round_terms['A']
        self.global_terms['B'] += self.round_terms['B']
        self.prev_lambda_reg    = self.lambda_reg

        # Reset round accumulators after integration
        self.round_terms['A'].fill(0)
        self.round_terms['B'].fill(0)

    def complete_aggregation_round(self) -> Dict[str, np.ndarray]:
        """
        Perform the global update of the model on the coordinating node.
        """
        t_fit = max(self.round_fit_times) if self.round_fit_times else 0.0
        start = time.perf_counter()

        # Δλ: change in regularization parameter since last round
        mu = self.lambda_reg - self.prev_lambda_reg 
        # U: accumulated local updates from current round
        U = self.round_terms['A'] + mu * _penalty_matrix(self.n_features + 1)
        # Snapshot of the previous global matrix
        P_prev = self.global_terms["P"]

        if self.diagnostics is not None:
            self.diagnostics.record_round_update(
                U = U,
                P_prev = P_prev,
                effective_rank_fn = self.diagnostics._effective_rank,
            )

        # Initialize or accumulate P
        self.global_terms["P"] = U if P_prev is None else P_prev + U

        # Eigendecomposition of ΔA (for Woodbury update)
        S, Q = eigh(U, check_finite = False)

        if self.numeric_checks:
            self._run_numeric_checks(U, S, Q)

        # Update the inverse of P using the Sherman-Morrison-Woodbury formula
        self.global_terms['P_inv'] = self.update_inverse(Q, S)

        # Aggregate the round contributions into the global A and B matrices
        # and reset temporary round accumulators
        self.update_global_matrices()

        # Compute the updated global model coefficients
        self.compute_global_model()

        self.round_elapsed += (time.perf_counter() - start) + t_fit

        if self.diagnostics is not None:
            self.diagnostics.finalize_round()

        return (self.global_terms['P_inv'], self.global_terms['B']) 
        
    def update_inverse(self, Q: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Update the inverse of the global matrix P using the Sherman-Morrison-Woodbury formula.
        """

        # Get the current inverse of P
        P_inv = self.global_terms['P_inv']

        # If P_inv does not exist yet, compute it from scratch
        if P_inv is None:
            try:
                # pinvh is suited to symmetric matrices
                P_inv = pinvh(self.global_terms['P'])
            except Exception as e:
                logger.warning(f"Failed to compute pinvh: {e}. Adding regularization and retrying.")
                # Retry with a small ridge term
                eps = 1e-6
                P_reg = self.global_terms['P'] + eps * np.eye(self.global_terms['P'].shape[0])
                P_inv = np.linalg.pinv(P_reg)
            
            return P_inv

        # 0. Keep only non-null eigen-directions of U (rank-rho update).
        #    Null/round-off directions must be DROPPED (1/sigma = inf), not set to 1/sigma = 0.
        tol = eps * max(1.0, float(np.max(np.abs(S))))
        keep = np.abs(S) > tol
        if not np.any(keep):
            return P_inv
        Q_r, S_r = Q[:, keep], S[keep]

        # 1. Capacitance matrix M = Sigma_r^{-1} + Q_r^T P^{-1} Q_r  (rho x rho)
        PQ = P_inv @ Q_r
        M = np.diag(1.0 / S_r) + Q_r.T @ PQ

        # 2. Woodbury: (P + Q_r S_r Q_r^T)^{-1} = P^{-1} - P^{-1}Q_r M^{-1} Q_r^T P^{-1}
        updated_P_inv = P_inv - PQ @ np.linalg.solve(M, PQ.T)
        return 0.5 * (updated_P_inv + updated_P_inv.T)  # keep symmetry

    def compute_global_model(self) -> np.ndarray:
        """
        Returns the current estimate of the global beta vector.
        """
        self.global_model = self.global_terms['P_inv'] @ self.global_terms['B']

        # Cheap O(d^2) check of the normal equations P w = B; refactorize only if Woodbury drifted
        P, B = self.global_terms['P'], self.global_terms['B']
        rel_res = np.linalg.norm(P @ self.global_model - B) / max(np.linalg.norm(B), 1e-300)
        # Attainable accuracy is ~eps*cond(P); cond_1 is estimated in O(d^2) from P and P^{-1}.
        # Refactorize only if the residual exceeds what the conditioning allows (real drift).
        cond = np.linalg.norm(P, 1) * np.linalg.norm(self.global_terms['P_inv'], 1)
        tol = max(self.residual_tol, 100 * np.finfo(float).eps * cond)
        if rel_res > tol:
            logger.warning(f"[{self.__class__.__name__}] Woodbury residual {rel_res:.2e}: recomputing inverse.")
            self.global_terms['P_inv'] = pinvh(P)
            self.global_model = self.global_terms['P_inv'] @ B
            self.n_refreshes += 1
