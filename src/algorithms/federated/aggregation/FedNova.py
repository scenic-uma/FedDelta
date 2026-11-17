import numpy as np
import logging

from .base import BaseAggregation

logger = logging.getLogger(__name__)

class FedNova(BaseAggregation):
    """
    Federated Normalized Averaging (FedNova) strategy.
    """
    def accumulate(
        self, 
        update: tuple[np.ndarray, int, float],
        total_samples: int,
        weighted_sum: np.ndarray | None,
        tau_weighted_sum: float | None
    ) -> None:
        """
        Aggregates weighted local updates from nodes.
        """
        # --------------------------
        # 1. Unpack update
        # --------------------------
        w_i, w_global, n_i, tau_i, _ = update
        tau_i = max(tau_i, 1)  # Guard against zero division

        # --------------------------
        # 2. Normalize to list
        # --------------------------
        if not isinstance(w_i, list):
            w_i = [np.asarray(w_i, dtype=np.float64)]
        else:
            w_i = [np.asarray(w, dtype=np.float64) for w in w_i]

        if w_global is not None:
            if not isinstance(w_global, list):
                w_global = [np.asarray(w_global, dtype=np.float64)]
            else:
                w_global = [np.asarray(w, dtype=np.float64) for w in w_global]
        
        # --------------------------
        # 3. Init accumulator
        # --------------------------
        if weighted_sum is None:
            weighted_sum = [np.zeros_like(w, dtype=np.float64) for w in w_i]
            tau_weighted_sum = 0.0
        elif tau_weighted_sum is None:
            tau_weighted_sum = 0.0

        if len(weighted_sum) != len(w_i):
            raise ValueError(
                f"[FedNova.accumulate] mismatch layers: "
                f"{len(weighted_sum)} vs {len(w_i)}"
            )
        
        # --------------------------
        # 4. FedNova logic per layer
        # --------------------------
        for i in range(len(w_i)):
            if w_global is None:
                delta = w_i[i]
            else:
                delta = (w_i[i] - w_global[i]) / tau_i
            weighted_sum[i] += delta * n_i
        
        total_samples += n_i
        tau_weighted_sum += n_i * tau_i
        
        return total_samples, weighted_sum, tau_weighted_sum
    
    def process_global_update(
        self, 
        global_model, 
        total_samples: int, 
        weighted_sum: list,
        tau_weighted_sum: float
    ) -> np.ndarray:
        """
        Applies the aggregated weighted update to the global model.
        """

        if total_samples == 0 or weighted_sum is None:
            logger.warning("[FedNova.process_global_update] Nothing to aggregate.")
            return global_model
        
        tau_eff = tau_weighted_sum / total_samples if total_samples > 0 else 1.0

        if hasattr(global_model, "set_weights"):
            # --- FedLSTM path ---
            global_weights = global_model.get_weights()

            if len(global_weights) != len(weighted_sum):
                raise ValueError(
                    f"[FedNova.process_global_update] layer mismatch: "
                    f"model has {len(global_weights)}, accumulator has {len(weighted_sum)}"
                )
            
            new_weights = []
            for g, acc in zip(global_weights, weighted_sum):
                normalized_delta = acc / total_samples
                update = tau_eff * normalized_delta          
                new_weights.append(g + update)

            global_model.set_weights(new_weights)
            return global_model

        else:
            # --- FedGD path: global_model es np.ndarray ---
            global_w = np.asarray(global_model, dtype=np.float64)

            # weighted_sum tiene un solo elemento (el vector de pesos)
            if len(weighted_sum) != 1:
                raise ValueError(
                    f"[FedNova.process_global_update] FedGD expects 1 layer, "
                    f"got {len(weighted_sum)}"
                )
            
            normalized_delta = weighted_sum[0] / total_samples
            update = tau_eff * normalized_delta

            new_weights = global_w + update.reshape(global_w.shape)

            return new_weights

    

