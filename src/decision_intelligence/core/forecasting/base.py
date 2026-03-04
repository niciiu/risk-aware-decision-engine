"""
Probabilistic forecasting abstractions.

ForecastResult is intentionally distribution-agnostic:
  - quantiles: covers any confidence interval without assuming Gaussian.
  - samples:   enables Monte-Carlo downstream (risk, simulation pipelines).
  - point:     best-estimate scalar/array (e.g. median or mean).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ForecastResult:
    """Distribution-agnostic probabilistic forecast output."""

    point: np.ndarray
    # keyed by quantile level string, e.g. {"0.05": array, "0.50": array, "0.95": array}
    quantiles: Dict[str, np.ndarray] = field(default_factory=dict)
    # raw posterior / bootstrap samples — shape: (n_samples, horizon)
    samples: Optional[np.ndarray] = None
    quantile_levels: List[float] = field(default_factory=lambda: [0.05, 0.50, 0.95])
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lower(self) -> Optional[np.ndarray]:
        if not self.quantiles:
            return None
        return self.quantiles.get(str(min(float(k) for k in self.quantiles)))

    @property
    def upper(self) -> Optional[np.ndarray]:
        if not self.quantiles:
            return None
        return self.quantiles.get(str(max(float(k) for k in self.quantiles)))

    @property
    def uncertainty(self) -> Optional[np.ndarray]:
        """Interval width — proxy for epistemic uncertainty."""
        if self.lower is not None and self.upper is not None:
            return self.upper - self.lower
        return None


class BaseForecaster(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseForecaster": ...

    @abstractmethod
    def predict(self, X: Any) -> ForecastResult: ...

    def validate_input(self, X: Any) -> None:
        """Optional: raise ValueError if X is invalid."""

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
