"""
Adaptive server optimizers (Reddi et al., "Adaptive Federated Optimization", ICLR 2021).

Clients run local SGD as in FedAvg. The server treats the weighted average
change Δ = avg(w_i) - x as a pseudo-gradient and applies:
  FedAvgM   : m = β1·m + Δ;                     x += η·m
  FedAdagrad: m = β1·m + (1-β1)Δ; v = v + Δ²;                       x += η·m / (√v + τ)
  FedAdam   : m = β1·m + (1-β1)Δ; v = β2·v + (1-β2)Δ²;           x += η·m / (√v + τ)
  FedYogi   : m as Adam; v = v - (1-β2)·Δ²·sign(v - Δ²);         x += η·m / (√v + τ)
v is initialised to τ² (Alg. 2: v_{-1} >= τ²) and the momentum m is shared by
FedAdagrad, FedYogi and FedAdam (Alg. 2, line 11). Δ is sample-weighted as in FedAvg
; server state (m, v) lives in the aggregator.
"""
import logging
import numpy as np

from .FedAvg import FedAvg

logger = logging.getLogger(__name__)


class FedOpt(FedAvg):
    variant = "adam"

    def __init__(self, lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.99, tau: float = 1e-3):
        self.lr, self.beta1, self.beta2, self.tau = lr, beta1, beta2, tau
        self.m = None
        self.v = None

    def process_global_update(self, global_model, total_samples, weighted_sum, tau_weighted_sum=None):
        if total_samples == 0 or weighted_sum is None:
            logger.warning("No updates received in this round")
            return global_model

        is_nn = hasattr(global_model, "set_weights")
        x = global_model.get_weights() if is_nn else [np.asarray(global_model, dtype=np.float64)]
        avg = [w / float(total_samples) for w in weighted_sum]
        delta = [a.reshape(xi.shape) - xi for a, xi in zip(avg, x)]

        if self.m is None:
            self.m = [np.zeros_like(d) for d in delta]
            self.v = [np.full_like(d, self.tau ** 2) for d in delta]

        new = []
        for k, (xi, d) in enumerate(zip(x, delta)):
            d2 = d * d
            if self.variant == "avgm":
                self.m[k] = self.beta1 * self.m[k] + d
                new.append(xi + self.lr * self.m[k])
                continue
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * d
            if self.variant == "adagrad":
                self.v[k] = self.v[k] + d2
            elif self.variant == "yogi":
                self.v[k] = self.v[k] - (1 - self.beta2) * d2 * np.sign(self.v[k] - d2)
            else:  # adam
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * d2
            new.append(xi + self.lr * self.m[k] / (np.sqrt(self.v[k]) + self.tau))

        if is_nn:
            global_model.set_weights(new)
            return global_model
        return new[0]


class FedAvgM(FedOpt):
    variant = "avgm"

    def __init__(self, lr: float = 1.0, beta1: float = 0.9, **kwargs):
        super().__init__(lr=lr, beta1=beta1, **kwargs)
