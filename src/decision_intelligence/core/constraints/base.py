"""
Constraint definition, evaluation, and registry.

Hard constraints -> must never be violated (feasibility gate).
Soft constraints -> penalised in the objective when violated.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ConstraintViolation:
    constraint_name: str
    severity: str         # "hard" | "soft"
    message: str
    value: Any = None
    threshold: Any = None
    penalty: float = 0.0  # soft constraints: amount added to objective


class BaseConstraint(ABC):
    name: str = "unnamed"
    severity: str = "hard"

    @abstractmethod
    def evaluate(self, solution: Any, context: Dict[str, Any]) -> bool:
        """Return True iff the constraint is satisfied."""
        ...

    def violation(self, solution: Any, context: Dict[str, Any]) -> Optional[ConstraintViolation]:
        if not self.evaluate(solution, context):
            return ConstraintViolation(self.name, self.severity, f"'{self.name}' violated.")
        return None


class ConstraintRegistry:
    def __init__(self) -> None:
        self._constraints: Dict[str, BaseConstraint] = {}

    def register(self, constraint: BaseConstraint) -> None:
        self._constraints[constraint.name] = constraint

    def evaluate_all(self, solution: Any, context: Dict[str, Any]) -> List[ConstraintViolation]:
        return [v for c in self._constraints.values() if (v := c.violation(solution, context))]

    def is_feasible(self, solution: Any, context: Dict[str, Any]) -> bool:
        """True only when no HARD constraints are violated."""
        return all(v.severity != "hard" for v in self.evaluate_all(solution, context))

    def total_penalty(self, solution: Any, context: Dict[str, Any]) -> float:
        """Sum of soft-constraint penalties — addable to objective value."""
        return sum(v.penalty for v in self.evaluate_all(solution, context) if v.severity == "soft")
