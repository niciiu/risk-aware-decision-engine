"""Objective function abstractions."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ObjectiveResult:
    value: float
    gradient: Optional[np.ndarray] = None
    component_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseObjective(ABC):
    name: str = "unnamed_objective"
    direction: str = "minimize"  # "minimize" | "maximize"

    @abstractmethod
    def evaluate(self, solution: Any, context: Dict[str, Any]) -> ObjectiveResult: ...

    def sign(self) -> float:
        """Return +1 for minimization, -1 for maximization (unified solvers)."""
        return 1.0 if self.direction == "minimize" else -1.0


class CompositeObjective(BaseObjective):
    """
    Weighted scalarisation of multiple objectives.

    Example
    -------
    obj = CompositeObjective(objectives=[(cost_obj, 0.70), (reliability_obj, 0.30)])
    """
    def __init__(self, objectives: List[Tuple[BaseObjective, float]], direction: str = "minimize") -> None:
        self.objectives = objectives
        self.direction = direction

    def evaluate(self, solution: Any, context: Dict[str, Any]) -> ObjectiveResult:
        total = 0.0
        components: Dict[str, float] = {}
        for obj, weight in self.objectives:
            result = obj.evaluate(solution, context)
            components[obj.name] = result.value
            total += weight * result.value
        return ObjectiveResult(value=total, component_values=components)
