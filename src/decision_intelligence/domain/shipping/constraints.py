"""
Shipping-domain constraints.

BudgetConstraint : hard  — total cost must not exceed budget_cap
SLAConstraint    : soft  — weighted avg lead time should meet threshold

Why implement penalty in violation() (not in objective):
    BaseConstraint.violation() is the single place penalty is computed.
    ConstraintRegistry.total_penalty() aggregates from there.
    Putting penalty logic anywhere else breaks the registry contract.

Business ownership:
    BudgetConstraint → Finance
    SLAConstraint    → Operations
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from decision_intelligence.core.constraints.base import (
    BaseConstraint,
    ConstraintViolation,
)
from decision_intelligence.domain.shipping.objective import VendorProfile


class BudgetConstraint(BaseConstraint):
    """
    Hard constraint: total expected cost must not exceed budget_cap.

    Math: Σ(x_i * cost_i) * demand ≤ budget_cap

    Why hard:
        Budget caps are Finance mandates.
        Violating them is a policy breach, not a tradeoff.
        System must declare infeasibility and stop — not find
        a "best effort" solution that exceeds budget.

    Slack interpretation:
        slack > 0 : budget has headroom
        slack = 0 : constraint exactly binding
        slack < 0 : infeasible by |slack| dollars
    """

    name = "budget_cap"
    severity = "hard"

    def __init__(
        self,
        vendors: List[VendorProfile],
        budget_cap: float,
    ) -> None:
        self.vendors = vendors
        self.budget_cap = budget_cap
        self._ids = [v.vendor_id for v in vendors]
        self._costs = np.array([v.cost_per_unit for v in vendors])

    def evaluate(self, solution: Any, context: Dict[str, Any]) -> bool:
        demand: float = context.get("features", context).get("demand", 1.0)
        x = np.array([solution.get(vid, 0.0) for vid in self._ids])
        total_cost = float(np.dot(x, self._costs) * demand)
        return total_cost <= self.budget_cap

    def violation(
        self, solution: Any, context: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        demand: float = context.get("features", context).get("demand", 1.0)
        x = np.array([solution.get(vid, 0.0) for vid in self._ids])
        total_cost = float(np.dot(x, self._costs) * demand)
        slack = self.budget_cap - total_cost

        if slack >= 0:
            return None

        return ConstraintViolation(
            constraint_name=self.name,
            severity=self.severity,
            message=(
                f"Total cost ${total_cost:,.0f} exceeds "
                f"budget_cap ${self.budget_cap:,.0f} by ${-slack:,.0f}. "
                f"Owner: Finance."
            ),
            value=total_cost,
            threshold=self.budget_cap,
            penalty=0.0,  # hard constraint: no penalty, solver enforces via bounds
        )


class SLAConstraint(BaseConstraint):
    """
    Soft constraint: demand-weighted avg lead time should not exceed threshold.

    Math:
        weighted_lead_time = Σ(x_i * lead_time_i)
        excess_days        = max(0, weighted_lead_time - sla_threshold_days)
        penalty            = excess_days * penalty_per_day * demand

    Why soft:
        SLA breaches are costly but sometimes unavoidable.
        System should make best feasible decision and flag the breach —
        not declare infeasibility and stop.
        Serving customer late > not serving them at all.

    Why penalty scales with demand:
        1-day delay on 10,000 units is 10x costlier than on 1,000 units.
        Proportionate penalty reflects real operational impact.
    """

    name = "sla_threshold"
    severity = "soft"

    def __init__(
        self,
        vendors: List[VendorProfile],
        sla_threshold_days: float,
        penalty_per_day: float,
    ) -> None:
        self.vendors = vendors
        self.sla_threshold_days = sla_threshold_days
        self.penalty_per_day = penalty_per_day
        self._ids = [v.vendor_id for v in vendors]
        self._lead_times = np.array([v.lead_time_days for v in vendors])

    def evaluate(self, solution: Any, context: Dict[str, Any]) -> bool:
        x = np.array([solution.get(vid, 0.0) for vid in self._ids])
        weighted_lead_time = float(np.dot(x, self._lead_times))
        return weighted_lead_time <= self.sla_threshold_days

    def violation(
        self, solution: Any, context: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        demand: float = context.get("features", context).get("demand", 1.0)
        x = np.array([solution.get(vid, 0.0) for vid in self._ids])
        weighted_lead_time = float(np.dot(x, self._lead_times))
        excess_days = max(0.0, weighted_lead_time - self.sla_threshold_days)

        if excess_days == 0.0:
            return None

        penalty = excess_days * self.penalty_per_day * demand

        return ConstraintViolation(
            constraint_name=self.name,
            severity=self.severity,
            message=(
                f"Weighted avg lead time {weighted_lead_time:.2f}d "
                f"exceeds SLA threshold {self.sla_threshold_days}d "
                f"by {excess_days:.2f}d. "
                f"Penalty: ${penalty:,.0f}. Owner: Operations."
            ),
            value=weighted_lead_time,
            threshold=self.sla_threshold_days,
            penalty=penalty,
        )