"""
ShippingObjective — concrete BaseObjective for vendor allocation.

Mathematical definition:
    objective(x) =
        w_cost * Σ(x_i * cost_i * demand)
      + w_sla  * Σ(x_i * max(0, lead_time_i - sla_threshold) * penalty_per_day * demand)
      + w_risk * Σ(x_i * (1 - reliability_i))

Where:
    x_i    = fraction of demand allocated to vendor i  (decision variable)
    demand = total units to fulfill (from context["demand"])

Why three terms:
    cost term  → Finance owns this. Minimize total spend.
    SLA term   → Operations owns this. Penalize slow vendors.
    risk term  → Risk team owns this. Penalize unreliable vendors.

Why weighted sum:
    Tractable, explainable. A VP can understand
    "we weight cost 60%, SLA 25%, risk 15%."
    CVaR objective comes in Phase 2 — same interface, different compute.

Business ownership of weights:
    Weights live in config — not in code.
    Business can tune without engineering involvement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from decision_intelligence.core.objective.base import BaseObjective, ObjectiveResult


@dataclass(frozen=True)
class VendorProfile:
    """
    Immutable vendor data used by the objective.

    Why frozen:
        Vendor properties do not change during a decision run.
        Mutability here would be a bug, not a feature.

    Why separate from domain Vendor model:
        Objective only needs cost, reliability, lead_time.
        Keeping it minimal prevents objective from depending
        on domain model changes (e.g. adding address field).
    """
    vendor_id: str
    cost_per_unit: float
    reliability: float   # P(on-time delivery) in [0, 1]
    lead_time_days: float


class ShippingObjective(BaseObjective):
    """
    Multi-objective function for continuous vendor allocation.

    Solution format expected:
        solution = {
            "vendor_A": 0.40,   # fraction of demand
            "vendor_B": 0.60,
            ...
        }
        Keys must match vendor_id in vendors list.
        Values must sum to 1.0 (enforced by solver equality constraint).

    Context keys used:
        demand          : float  — total units to fulfill
        (all others ignored — objective is context-minimal by design)

    Phase 2 extension:
        Add mode="cvar" parameter.
        When cvar, replace cost term with CVaR estimate from Monte Carlo.
        Interface (evaluate signature) stays identical.
    """

    name = "shipping_multi_objective"
    direction = "minimize"

    def __init__(
        self,
        vendors: List[VendorProfile],
        w_cost: float,
        w_sla: float,
        w_risk: float,
        sla_threshold_days: float,
        penalty_per_day: float,
    ) -> None:
        """
        Args:
            vendors            : ordered list of vendor profiles
            w_cost             : weight for cost term (Finance)
            w_sla              : weight for SLA penalty term (Operations)
            w_risk             : weight for reliability risk term (Risk team)
            sla_threshold_days : delivery time beyond this triggers penalty
            penalty_per_day    : $ per unit per day beyond SLA threshold
        """
        self.vendors = vendors
        self.w_cost = w_cost
        self.w_sla = w_sla
        self.w_risk = w_risk
        self.sla_threshold_days = sla_threshold_days
        self.penalty_per_day = penalty_per_day

        # Pre-compute arrays for vectorized ops — called thousands of times by solver
        self._ids = [v.vendor_id for v in vendors]
        self._costs = np.array([v.cost_per_unit for v in vendors])
        self._reliabilities = np.array([v.reliability for v in vendors])
        self._lead_times = np.array([v.lead_time_days for v in vendors])

    def _allocation_vector(self, solution: Dict[str, Any]) -> np.ndarray:
        """
        Extract ordered allocation array from solution dict.

        Why ordered:
            Solver works with numpy arrays indexed by position.
            Dict ordering must match vendor list ordering.
            This method is the single place that enforces that mapping.
        """
        return np.array([solution.get(vid, 0.0) for vid in self._ids])

    def _cost_term(self, x: np.ndarray, demand: float) -> float:
        """
        Total expected shipping cost.

        Math: Σ(x_i * cost_i) * demand
        Unit: dollars

        Why scale by demand:
            x_i is a fraction. Multiplying by demand gives absolute spend.
            Keeps cost term in same unit space as SLA penalty term.
        """
        return float(np.dot(x, self._costs) * demand)

    def _sla_penalty_term(self, x: np.ndarray, demand: float) -> float:
        """
        Expected SLA penalty from late deliveries.

        Math: Σ(x_i * max(0, lead_time_i - threshold) * penalty_per_day) * demand
        Unit: dollars

        Why max(0, ...):
            Vendors meeting SLA contribute zero penalty.
            SLA is a floor, not a target — fast vendors get no reward here.
        """
        overdue = np.maximum(0.0, self._lead_times - self.sla_threshold_days)
        return float(np.dot(x, overdue) * self.penalty_per_day * demand)

    def _risk_term(self, x: np.ndarray) -> float:
        """
        Reliability-weighted failure exposure.

        Math: Σ(x_i * (1 - reliability_i))
        Unit: dimensionless [0, 1]

        Why not scale by demand:
            Risk is a rate, not an absolute quantity.
            10% failure rate is 10% regardless of volume.

        Phase 2 note:
            This becomes E[P(failure)] under demand distribution.
            Math stays identical — only x becomes E[x] over samples.
        """
        return float(np.dot(x, 1.0 - self._reliabilities))

    def evaluate(
        self,
        solution: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ObjectiveResult:
        """
        Compute weighted objective + full component breakdown.

        Returns ObjectiveResult with component_values populated —
        this is what powers the audit trail and interview explanation:
        "why was vendor D preferred over vendor B?"
        """
        demand: float = context.get("features", context).get("demand", 1.0)

        x = self._allocation_vector(solution)

        cost = self._cost_term(x, demand)
        sla_penalty = self._sla_penalty_term(x, demand)
        risk = self._risk_term(x)

        weighted_total = (
            self.w_cost * cost
            + self.w_sla * sla_penalty
            + self.w_risk * risk
        )

        return ObjectiveResult(
            value=weighted_total,
            component_values={
                "cost_term": cost,
                "sla_penalty_term": sla_penalty,
                "risk_term": risk,
                "weighted_total": weighted_total,
            },
            metadata={
                "demand": demand,
                "weights": {
                    "w_cost": self.w_cost,
                    "w_sla": self.w_sla,
                    "w_risk": self.w_risk,
                },
            },
        )
