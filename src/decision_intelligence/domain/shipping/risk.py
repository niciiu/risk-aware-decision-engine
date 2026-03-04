"""
ShippingRiskEvaluator — shipping-domain Monte Carlo risk evaluator.

Extends MonteCarloRiskEvaluator with shipping-specific cost calculation.

Why separate from MonteCarloRiskEvaluator:
    Core evaluator is domain-agnostic — it handles simulation mechanics.
    This class only knows shipping math: how to compute cost and SLA breach
    given a vendor allocation and a demand sample.

    Tomorrow, add WarehouseRiskEvaluator with different cost formula —
    MonteCarloRiskEvaluator stays untouched.

Business ownership:
    Cost formula → Finance (owns unit economics)
    SLA formula  → Operations (owns delivery standards)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from decision_intelligence.core.risk.monte_carlo import (
    DemandDistribution,
    MonteCarloRiskEvaluator,
)
from decision_intelligence.domain.shipping.objective import VendorProfile


class ShippingRiskEvaluator(MonteCarloRiskEvaluator):
    """
    Evaluates risk of a vendor allocation under demand uncertainty.

    Cost model:
        cost(x, D) = Σ(x_i * cost_i * D) + sla_penalty(x, D)

    Where:
        x_i    = allocation fraction for vendor i
        cost_i = cost per unit for vendor i
        D      = sampled demand (varies per simulation)

    SLA penalty:
        excess_days = max(0, weighted_lead_time - sla_threshold)
        penalty     = excess_days * penalty_per_day * D

    Why demand enters linearly:
        Both shipping cost and SLA penalty scale with volume.
        Doubling demand doubles cost and penalty — linear assumption
        is valid for commodity shipping at typical volumes.

    Phase 3 extension:
        Add lead_time uncertainty: lead_time_i ~ LogNormal(mu_i, sigma_i)
        Sample jointly with demand for correlated stress scenarios.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vendors: List[VendorProfile],
        demand_distribution: DemandDistribution,
        sla_threshold_days: float,
        penalty_per_day: float,
    ) -> None:
        super().__init__(config=config, demand_distribution=demand_distribution)
        self.vendors = vendors
        self.sla_threshold_days = sla_threshold_days
        self.penalty_per_day = penalty_per_day

        # Pre-compute arrays — called N times in simulation loop
        self._ids = [v.vendor_id for v in vendors]
        self._costs = np.array([v.cost_per_unit for v in vendors])
        self._lead_times = np.array([v.lead_time_days for v in vendors])

    def _allocation_vector(self, solution: Any) -> np.ndarray:
        return np.array([solution.get(vid, 0.0) for vid in self._ids])

    def compute_cost(
        self,
        solution: Any,
        demand: float,
        context: Dict[str, Any],
    ) -> float:
        """
        Total cost = shipping cost + SLA penalty for one demand sample.

        Why include SLA penalty in cost:
            SLA breach has a real dollar cost (contractual or reputational).
            Excluding it understates true cost in bad scenarios.
            CVaR over total cost is more meaningful than CVaR over raw shipping cost.
        """
        x = self._allocation_vector(solution)
        shipping_cost = float(np.dot(x, self._costs) * demand)
        overdue = np.maximum(0.0, self._lead_times - self.sla_threshold_days)
        sla_penalty = float(np.dot(x, overdue) * self.penalty_per_day * demand)
        return shipping_cost + sla_penalty

    def compute_sla_breach(
        self,
        solution: Any,
        demand: float,
        context: Dict[str, Any],
    ) -> bool:
        """
        True if weighted avg lead time exceeds SLA threshold.

        Note: SLA breach is independent of demand volume —
        it depends only on allocation fractions and lead times.
        Demand parameter kept for interface consistency and
        future extension (e.g. demand-dependent lead times in Phase 3).
        """
        x = self._allocation_vector(solution)
        weighted_lead = float(np.dot(x, self._lead_times))
        return weighted_lead > self.sla_threshold_days