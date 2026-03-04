"""Risk quantification abstractions (VaR, CVaR, drawdown, volatility)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RiskReport:
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    is_within_tolerance: bool = True
    extra_metrics: Dict[str, float] = field(default_factory=dict)


class BaseRiskModel(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def assess(self, solution: Any, context: Dict[str, Any]) -> RiskReport: ...
