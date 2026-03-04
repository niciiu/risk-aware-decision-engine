"""
Solver abstractions.

solve() is explicit about all inputs so the solver is composable
independently of the orchestrator — no hidden state coupling.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from decision_intelligence.core.constraints.base import ConstraintRegistry
from decision_intelligence.core.objective.base import BaseObjective


@dataclass
class SearchSpace:
    """Defines the feasible domain for the solver to explore."""
    bounds: Dict[str, Tuple[float, float]]   # {var_name: (lower, upper)}
    discrete_vars: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    best_solution: Any
    best_value: float
    converged: bool
    iterations: int
    is_feasible: bool = True
    constraint_violations: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSolver(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def solve(
        self,
        objective: BaseObjective,
        constraints: ConstraintRegistry,
        search_space: SearchSpace,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """
        Parameters
        ----------
        objective    : what to optimise (knows its own direction)
        constraints  : hard + soft constraints to enforce / penalise
        search_space : variable bounds and discrete structure
        context      : runtime data (forecasts, regime, features, ...)
        """
        ...
