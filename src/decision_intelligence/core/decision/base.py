"""
Decision engine base — optimization-first, not classification-first.

A DecisionEngine owns:
  - an Objective         (what to optimise)
  - a ConstraintRegistry (what must hold)
  - a RiskModel          (risk-awareness)
  - a Solver             (how to search)

It contains NO domain logic — that lives in domain adapters.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from decision_intelligence.core.constraints.base import ConstraintRegistry
from decision_intelligence.core.objective.base import BaseObjective
from decision_intelligence.core.optimization.base import BaseSolver, OptimizationResult, SearchSpace
from decision_intelligence.core.risk.base import BaseRiskModel


@dataclass
class DecisionContext:
    """All runtime information needed to make a decision."""
    timestamp: str
    domain: str
    features: Dict[str, Any] = field(default_factory=dict)
    forecast: Optional[Any] = None   # ForecastResult — optional at base level
    regime: Optional[Any] = None     # RegimeState    — optional at base level
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutput:
    """Structured result: winning action + full optimisation trace."""
    action: str
    confidence: float
    rationale: str
    optimization_result: Optional[OptimizationResult] = None
    risk_report: Optional[Any] = None
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDecisionEngine(ABC):
    """
    Abstract optimization-aware decision engine.

    Subclasses inject concrete implementations at construction time.
    The engine is stateless per call — all runtime state lives in context.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        objective: Optional[BaseObjective] = None,
        constraints: Optional[ConstraintRegistry] = None,
        risk_model: Optional[BaseRiskModel] = None,
        solver: Optional[BaseSolver] = None,
    ) -> None:
        self.config = config
        self.objective = objective
        self.constraints = constraints or ConstraintRegistry()
        self.risk_model = risk_model
        self.solver = solver

    @abstractmethod
    def build_search_space(self, context: DecisionContext) -> SearchSpace:
        """Define the variable bounds for this decision problem."""
        ...

    @abstractmethod
    def solution_to_action(self, solution: Any, context: DecisionContext) -> str:
        """Translate the raw optimiser output into a domain action string."""
        ...

    def decide(self, context: DecisionContext) -> DecisionOutput:
        """
        Standard decision loop.
        Subclasses override build_search_space() + solution_to_action() only.
        """
        if self.objective is None or self.solver is None:
            raise RuntimeError("objective and solver must be set before calling decide().")

        search_space = self.build_search_space(context)
        opt_result = self.solver.solve(
            objective=self.objective,
            constraints=self.constraints,
            search_space=search_space,
            context=context.__dict__,
        )

        risk_report = None
        if self.risk_model is not None:
            risk_report = self.risk_model.assess(
                solution=opt_result.best_solution,
                context=context.__dict__,
            )

        action = self.solution_to_action(opt_result.best_solution, context)

        return DecisionOutput(
            action=action,
            confidence=float(opt_result.converged),
            rationale=(
                f"Solver converged={opt_result.converged} "
                f"in {opt_result.iterations} iterations, "
                f"best_value={opt_result.best_value:.4f}"
            ),
            optimization_result=opt_result,
            risk_report=risk_report,
        )

    def explain(self, output: DecisionOutput) -> str:
        return output.rationale
