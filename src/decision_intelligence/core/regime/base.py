"""Regime detection abstractions."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RegimeState:
    label: str
    probability: float
    features: Dict[str, Any] = field(default_factory=dict)


class BaseRegimeDetector(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def fit(self, X: Any) -> "BaseRegimeDetector": ...

    @abstractmethod
    def detect(self, X: Any) -> List[RegimeState]: ...

    def dominant_regime(self, X: Any) -> RegimeState:
        """Convenience: return the single most probable regime."""
        return max(self.detect(X), key=lambda r: r.probability)
