import logging
import numpy as np

from .base import BaseAggregation

logger = logging.getLogger(__name__)

class FedAvg(BaseAggregation):
    """
    Federated Averaging (FedAvg) strategy.
    """  
    def accumulate(
        self, 
        update: tuple[np.ndarray, int],
        total_samples: int,
        weighted_sum: list[np.ndarray] | None,
        tau_weighted_sum: np.ndarray | None
    ):
        """
        Aggregates local model updates using a weighted average
        Compatible with:
        - Linear models (single ndarray)
        - Neural networks (list of ndarrays)
        """
        # --------------------------
        # 1. Unpack update
        # --------------------------
        w_i, _, n_i, _, _ = update
        # --------------------------
        # 2. Normalize to list
        # --------------------------
        if not isinstance(w_i, list):
            w_i = [w_i]

        # --------------------------
        # 3. Init accumulator
        # --------------------------
        if weighted_sum is None:
            weighted_sum = [np.zeros_like(w, dtype=np.float64) for w in w_i]

        # EXTRA safety
        if len(weighted_sum) != len(w_i):
            raise ValueError(
                f"[FedAvg.accumulate] mismatch layers: "
                f"{len(weighted_sum)} vs {len(w_i)}"
            )

        # --------------------------
        # 4. Accumulate
        # --------------------------
        for i in range(len(w_i)):
            if weighted_sum[i].shape != w_i[i].shape:
                raise ValueError(
                    f"[FedAvg.accumulate] shape mismatch at layer {i}: "
                    f"{weighted_sum[i].shape} vs {w_i[i].shape}"
                )

            weighted_sum[i] += np.asarray(w_i[i], dtype=np.float64) * n_i

        total_samples += n_i

        return total_samples, weighted_sum, tau_weighted_sum


    def process_global_update(
        self, 
        global_model, 
        total_samples: int, 
        weighted_sum: list,
        tau_weighted_sum: float = None

    ) -> np.ndarray:
        """
        Computes the final weighted average and updates the global model.
        """
        # If no samples have been accumulated, return the current model
        if total_samples == 0 or weighted_sum is None:
            logger.warning("No updates received in this round")
            return global_model
        
        # --------------------------
        # CASE 1: Keras / neural network model
        # --------------------------
        if hasattr(global_model, "set_weights"):
            # Weighted average (each element of weighted_sum divided by total_samples)
            averaged_weights = [(w_sum / float(total_samples)).astype(np.float32) for w_sum in weighted_sum]

            # Debug: check that the shapes match before setting weights
            gw = global_model.get_weights()
            if len(gw) != len(averaged_weights):
                raise ValueError(f"[FedAvg.process_global_update] mismatch: global expects {len(gw)} weight arrays but averaged has {len(averaged_weights)}")

            for i, (g_shape, a_shape) in enumerate(zip([w.shape for w in gw], [w.shape for w in averaged_weights])):
                if g_shape != a_shape:
                    raise ValueError(f"[FedAvg.process_global_update] shape mismatch at idx {i}: global {g_shape} vs averaged {a_shape}")
            # Assign weights to the global model
            global_model.set_weights(averaged_weights)
        
        # --------------------------
        # CASE 2: Regression
        # --------------------------
        else:
            # Compute the new global model as the weighted average
            global_model = weighted_sum[0] / total_samples
        
        return global_model