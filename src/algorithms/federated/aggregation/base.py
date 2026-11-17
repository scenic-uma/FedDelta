from abc import ABC, abstractmethod
import numpy as np

class BaseAggregation(ABC):

    @abstractmethod
    def accumulate(
        self,
        update,
        total_samples: int,
        weighted_sum: np.ndarray | None
    ) -> tuple[int, np.ndarray | None]:
        pass

    @abstractmethod
    def process_global_update(
        self,
        global_model: np.ndarray,
        total_samples: int,
        weighted_sum: np.ndarray | None
    ) -> np.ndarray:
        pass