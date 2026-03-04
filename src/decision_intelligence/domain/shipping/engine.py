"""
ShippingVendorEngine — concrete DecisionEngine for vendor allocation.

Implements the two domain-specific hooks only:
    - build_search_space()   : defines allocation bounds per vendor
    - solution_to_action()   : translates allocation dict to human-readable string

All optimization logic lives in the solver.
All objective logic lives in ShippingObjective.
This engine contains NO math — only domain translation.

Business ownership:
    This file is owned by the Logistics/Supply Chain domain team.
    They define what a valid search space looks like.
    Engineering owns the solver. Finance owns the objective weights.

Failure mode prevented:
    Without this separation, domain logic bleeds into the solver —
    making it impossible to reuse the solver for other domains.

Phase 2 extension:
    build_search_space() gains forecast-conditioned demand bounds
    when context.forecast is not None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from decision_intelligence.core.decision.base import (
    BaseDecisionEngine,
    DecisionContext,
    DecisionOutput,
)
from decision_intelligence.core.optimization.base import (
    OptimizationResult,
    SearchSpace,
)
from decision_intelligence.domain.shipping.objective import VendorProfile


@dataclass(frozen=True)
class VendorConfig:
    """
    Full vendor configuration including capacity.

    Separate from VendorProfile (used by objective) because:
        - Objective only needs cost, reliability, lead_time
        - Engine needs capacity_fraction to build search space
        - Separation prevents objective from depending on engine concerns
    """
    vendor_id: str
    cost_per_unit: float
    reliability: float
    lead_time_days: float
    capacity_fraction: float  # max fraction of demand this vendor can handle


class ShippingVendorEngine(BaseDecisionEngine):
    """
    Concrete engine for multi-vendor continuous allocation.

    Decision variable:
        x_i ∈ [0, capacity_fraction_i] for each vendor i
        Subject to: Σ x_i = 1  (enforced by solver)

    Why continuous allocation (not pick-one):
        Real procurement splits orders across vendors for:
        - Risk diversification
        - Capacity compliance
        - Cost-reliability tradeoff
        This is more realistic and analytically richer than argmin(cost).

    Construction:
        engine = ShippingVendorEngine(
            config={...},
            vendors=[VendorConfig(...)],
            objective=ShippingObjective(...),
            constraints=registry,
            solver=MixedScipySolver(config={...}),
        )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vendors: List[VendorConfig],
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.vendors = vendors

    def build_search_space(self, context: DecisionContext) -> SearchSpace:
        """
        Build allocation bounds from vendor capacity fractions.

        Bounds: x_i ∈ [0, capacity_fraction_i]
            Lower bound 0: vendor can be excluded entirely.
            Upper bound capacity_fraction_i: vendor cannot exceed their limit.

        Metadata passed to solver:
            budget_cap   : hard budget ceiling from config
            demand       : units to fulfill from context features
            cost_coeffs  : cost_per_unit per vendor (same order as bounds)

        Why metadata (not constraints):
            budget_cap is a hard constraint enforced by solver's
            inequality constraint — not a penalty. Solver needs the
            raw coefficient to build the scipy constraint function.
            ConstraintRegistry handles evaluation + audit trail separately.
        """
        demand: float = context.features.get("demand", 1.0)
        budget_cap: float = self.config.get("budget_cap", float("inf"))

        bounds: Dict[str, Any] = {
            v.vendor_id: (0.0, v.capacity_fraction)
            for v in self.vendors
        }

        return SearchSpace(
            bounds=bounds,
            discrete_vars=[],  # pure continuous allocation problem
            metadata={
                "budget_cap": budget_cap,
                "demand": demand,
                "cost_coeffs": [v.cost_per_unit for v in self.vendors],
                "vendor_ids": [v.vendor_id for v in self.vendors],
            },
        )

    def solution_to_action(
        self,
        solution: Optional[Dict[str, Any]],
        context: DecisionContext,
    ) -> str:
        """
        Translate allocation dict to human-readable decision string.

        Why this exists:
            Orchestrator and audit log need a string action.
            Raw solution dict is not human-readable for stakeholders.
            This is the single place that defines "what did the system decide."

        Infeasible case:
            Returns explicit infeasibility message — never silent failure.
        """
        if solution is None:
            return (
                "INFEASIBLE: No allocation satisfies all hard constraints. "
                f"budget_cap={self.config.get('budget_cap')}, "
                f"demand={context.features.get('demand')}. "
                "Action required: increase budget or reduce demand."
            )

        demand: float = context.features.get("demand", 1.0)

        lines = ["Vendor Allocation Decision:"]
        total_cost = 0.0
        vendor_map = {v.vendor_id: v for v in self.vendors}

        for vid, fraction in sorted(solution.items(), key=lambda x: -x[1]):
            if fraction < 1e-4:
                continue
            vendor = vendor_map.get(vid)
            if vendor is None:
                continue
            units = fraction * demand
            cost = units * vendor.cost_per_unit
            total_cost += cost
            lines.append(
                f"  {vid}: {fraction:.1%} "
                f"({units:,.0f} units, ${cost:,.0f})"
            )

        lines.append(f"  Total cost: ${total_cost:,.0f}")
        return "\n".join(lines)

    def decide(self, context: DecisionContext) -> DecisionOutput:
        """
        Override decide() to handle infeasible case explicitly.

        Why override:
            BaseDecisionEngine.decide() calls solution_to_action(None)
            when solver returns is_feasible=False — which is correct.
            But we also want infeasibility reflected in DecisionOutput.confidence.

        Infeasibility is a result, not an exception.
        """
        output = super().decide(context)

        if (
            output.optimization_result is not None
            and not output.optimization_result.is_feasible
        ):
            output.metadata["is_feasible"] = False
            output.metadata["infeasibility_reason"] = (
                "All solver restarts failed to find an allocation satisfying "
                "all hard constraints. See constraint_violations for details."
            )

        return output