"""
Decision Explainability — human-readable reasoning for allocation decisions.

Purpose:
    Optimization solvers are black boxes to stakeholders.
    This module translates solver output into structured explanations:
    - Why was each vendor selected (or excluded)?
    - Which objective component drove the decision?
    - Which constraints were active vs. slack?
    - How sensitive is the decision to weight changes?

Design principle:
    Explainability is a governance concern, not a solver concern.
    The solver optimizes. This module explains what the solver did.
    These are separate responsibilities with separate owners:
        Solver        → Engineering
        Explainability → Risk / Compliance / Business

Business value:
    Enterprise procurement decisions require audit trails.
    "The algorithm decided" is not acceptable to a CFO or regulator.
    This module produces the narrative that makes decisions defensible.

Integration:
    Works with any OptimizationResult + objective breakdown.
    Designed to feed into AuditLogger via structured dict output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VendorExplanation:
    """
    Per-vendor explanation of why it was included or excluded.

    Fields:
        vendor_id        : vendor identifier
        allocation       : fraction assigned (0.0 = excluded)
        cost_contribution: dollar cost from this vendor
        sla_status       : "within_sla" | "breaches_sla"
        reliability_risk : failure exposure from this vendor (rate)
        inclusion_reason : human-readable string explaining the decision
    """
    vendor_id: str
    allocation: float
    cost_contribution: float
    sla_status: str
    reliability_risk: float
    inclusion_reason: str


@dataclass
class ConstraintExplanation:
    """
    Explanation of a single constraint's status in the solution.

    Fields:
        name      : constraint identifier
        severity  : "hard" | "soft"
        status    : "satisfied" | "violated" | "active" (at boundary)
        slack     : distance from constraint boundary (positive = satisfied)
        narrative : plain-English explanation
    """
    name: str
    severity: str
    status: str
    slack: float
    narrative: str


@dataclass
class ObjectiveExplanation:
    """
    Breakdown of the objective function at the optimal solution.

    Shows which term dominated the decision — useful for weight tuning.
    """
    weighted_total: float
    cost_term: float
    cost_weight: float
    cost_contribution_pct: float      # % of weighted total from cost term
    sla_term: float
    sla_weight: float
    sla_contribution_pct: float
    risk_term: float
    risk_weight: float
    risk_contribution_pct: float
    dominant_term: str                # which term had highest weighted contribution


@dataclass
class SensitivityExplanation:
    """
    How much the objective changes per unit change in each weight.

    Interpretation:
        High sensitivity → weight has strong influence on decision.
        Low sensitivity  → weight is effectively irrelevant at this solution.

    Used by business to understand which levers matter most.
    """
    d_objective_d_w_cost: float    # ∂objective/∂w_cost
    d_objective_d_w_sla: float     # ∂objective/∂w_sla
    d_objective_d_w_risk: float    # ∂objective/∂w_risk
    most_sensitive_weight: str     # which weight has highest absolute sensitivity
    narrative: str


@dataclass
class DecisionExplanation:
    """
    Complete structured explanation of a vendor allocation decision.

    This is the primary output of DecisionExplainer.
    Can be serialized to dict for AuditLogger or rendered as text for humans.
    """
    decision_id: str
    demand: float
    total_cost: float
    vendors: List[VendorExplanation]
    objective: ObjectiveExplanation
    constraints: List[ConstraintExplanation]
    sensitivity: SensitivityExplanation
    summary: str                        # one-paragraph plain-English summary
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for AuditLogger integration."""
        return {
            "decision_id": self.decision_id,
            "demand": self.demand,
            "total_cost": self.total_cost,
            "vendors": [v.__dict__ for v in self.vendors],
            "objective": self.objective.__dict__,
            "constraints": [c.__dict__ for c in self.constraints],
            "sensitivity": self.sensitivity.__dict__,
            "summary": self.summary,
            **self.extra,
        }

    def to_text(self) -> str:
        """
        Render as human-readable text for reports and audit trails.
        Designed to be readable by a non-technical stakeholder.
        """
        lines = []
        lines.append("=" * 65)
        lines.append("  DECISION EXPLANATION")
        lines.append("=" * 65)
        lines.append(f"\n  Decision ID : {self.decision_id}")
        lines.append(f"  Demand      : {self.demand:,.0f} units")
        lines.append(f"  Total cost  : ${self.total_cost:,.0f}")

        # Vendor breakdown
        lines.append("\n" + "-" * 65)
        lines.append("  VENDOR ALLOCATION RATIONALE")
        lines.append("-" * 65)
        for v in sorted(self.vendors, key=lambda x: -x.allocation):
            if v.allocation < 1e-4:
                status = "  ✗ EXCLUDED"
            else:
                status = f"  ✓ {v.allocation:.1%}"
            lines.append(f"\n{status}  {v.vendor_id}")
            lines.append(f"          Cost contribution : ${v.cost_contribution:,.0f}")
            lines.append(f"          SLA status        : {v.sla_status}")
            lines.append(f"          Reliability risk  : {v.reliability_risk:.1%} failure exposure")
            lines.append(f"          Reason            : {v.inclusion_reason}")

        # Objective decomposition
        lines.append("\n" + "-" * 65)
        lines.append("  OBJECTIVE DECOMPOSITION")
        lines.append("-" * 65)
        lines.append(f"  Weighted total : {self.objective.weighted_total:,.4f}")
        lines.append(f"  Cost term      : {self.objective.cost_term:,.2f} × {self.objective.cost_weight} = {self.objective.cost_contribution_pct:.1f}% of objective")
        lines.append(f"  SLA term       : {self.objective.sla_term:,.2f} × {self.objective.sla_weight} = {self.objective.sla_contribution_pct:.1f}% of objective")
        lines.append(f"  Risk term      : {self.objective.risk_term:.4f} × {self.objective.risk_weight} = {self.objective.risk_contribution_pct:.1f}% of objective")
        lines.append(f"  Dominant term  : {self.objective.dominant_term}")

        # Constraint status
        lines.append("\n" + "-" * 65)
        lines.append("  CONSTRAINT STATUS")
        lines.append("-" * 65)
        for c in self.constraints:
            icon = "✅" if c.status == "satisfied" else "⚠️ " if c.severity == "soft" else "❌"
            lines.append(f"  {icon} [{c.severity.upper()}] {c.name} — {c.status.upper()}")
            lines.append(f"     Slack: {c.slack:+.4f}")
            lines.append(f"     {c.narrative}")

        # Sensitivity
        lines.append("\n" + "-" * 65)
        lines.append("  WEIGHT SENSITIVITY")
        lines.append("-" * 65)
        lines.append(f"  ∂obj/∂w_cost : {self.sensitivity.d_objective_d_w_cost:,.2f}")
        lines.append(f"  ∂obj/∂w_sla  : {self.sensitivity.d_objective_d_w_sla:,.2f}")
        lines.append(f"  ∂obj/∂w_risk : {self.sensitivity.d_objective_d_w_risk:.4f}")
        lines.append(f"  Most sensitive weight : {self.sensitivity.most_sensitive_weight}")
        lines.append(f"  {self.sensitivity.narrative}")

        # Summary
        lines.append("\n" + "-" * 65)
        lines.append("  SUMMARY")
        lines.append("-" * 65)
        lines.append(f"  {self.summary}")
        lines.append("=" * 65)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class DecisionExplainer:
    """
    Produces structured explanations for vendor allocation decisions.

    Usage:
        explainer = DecisionExplainer(
            vendor_profiles=vendors,
            w_cost=0.60, w_sla=0.25, w_risk=0.15,
            sla_threshold_days=3.0,
            penalty_per_day=1.5,
            budget_cap=55_000.0,
        )
        explanation = explainer.explain(
            solution={"vendor_A": 0.4, "vendor_D": 0.6},
            demand=10_000.0,
            decision_id="run_001",
        )
        print(explanation.to_text())
        audit_logger.log(AuditRecord(..., outputs=explanation.to_dict()))

    Why not subclass BaseExplainer:
        BaseExplainer.explain(model, input_data) is too generic.
        This class is domain-specific by design — it knows about
        vendors, costs, SLA, and risk. Generic interface would
        force awkward adapter patterns with no benefit.
    """

    def __init__(
        self,
        vendor_profiles: List[Any],       # List[VendorProfile] — avoid circular import
        w_cost: float,
        w_sla: float,
        w_risk: float,
        sla_threshold_days: float,
        penalty_per_day: float,
        budget_cap: float,
    ) -> None:
        self.vendors = vendor_profiles
        self.w_cost = w_cost
        self.w_sla = w_sla
        self.w_risk = w_risk
        self.sla_threshold_days = sla_threshold_days
        self.penalty_per_day = penalty_per_day
        self.budget_cap = budget_cap
        self._vendor_map = {v.vendor_id: v for v in vendor_profiles}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        solution: Dict[str, float],
        demand: float,
        decision_id: str = "unknown",
    ) -> DecisionExplanation:
        """
        Produce full explanation for a given allocation and demand.

        Args:
            solution    : allocation dict {vendor_id: fraction}
            demand      : total units to fulfill
            decision_id : run identifier for audit trail

        Returns:
            DecisionExplanation — structured, serializable, human-readable
        """
        vendor_explanations = self._explain_vendors(solution, demand)
        objective_explanation = self._explain_objective(solution, demand)
        constraint_explanations = self._explain_constraints(solution, demand)
        sensitivity_explanation = self._explain_sensitivity(solution, demand)

        total_cost = sum(v.cost_contribution for v in vendor_explanations)
        summary = self._build_summary(
            solution, demand, total_cost,
            objective_explanation, constraint_explanations,
        )

        return DecisionExplanation(
            decision_id=decision_id,
            demand=demand,
            total_cost=total_cost,
            vendors=vendor_explanations,
            objective=objective_explanation,
            constraints=constraint_explanations,
            sensitivity=sensitivity_explanation,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _explain_vendors(
        self,
        solution: Dict[str, float],
        demand: float,
    ) -> List[VendorExplanation]:
        """
        For each vendor, explain why it was included or excluded.

        Inclusion logic:
            - Vendors with allocation > 0 are "included"
            - Reason is determined by their SLA and cost rank
            - Excluded vendors get an explanation of why they lost out
        """
        explanations = []

        # Rank vendors by cost for reference
        cost_rank = sorted(
            self._vendor_map.keys(),
            key=lambda vid: self._vendor_map[vid].cost_per_unit,
        )

        for vendor_id, vendor in self._vendor_map.items():
            allocation = solution.get(vendor_id, 0.0)
            cost_contribution = allocation * vendor.cost_per_unit * demand
            reliability_risk = allocation * (1.0 - vendor.reliability)

            # SLA status
            if vendor.lead_time_days <= self.sla_threshold_days:
                sla_status = "within_sla"
            else:
                excess = vendor.lead_time_days - self.sla_threshold_days
                sla_status = f"breaches_sla_by_{excess:.1f}d"

            # Inclusion reason
            if allocation < 1e-4:
                if vendor.lead_time_days > self.sla_threshold_days:
                    reason = (
                        f"Excluded — lead time {vendor.lead_time_days}d exceeds "
                        f"SLA threshold {self.sla_threshold_days}d, "
                        f"incurring ${vendor.lead_time_days - self.sla_threshold_days:.1f}d × "
                        f"${self.penalty_per_day}/unit/day SLA penalty."
                    )
                elif vendor.reliability < 0.85:
                    reason = (
                        f"Excluded — reliability {vendor.reliability:.0%} below "
                        f"acceptable threshold. Risk exposure too high relative to cost savings."
                    )
                else:
                    cost_pos = cost_rank.index(vendor_id) + 1
                    reason = (
                        f"Excluded — cost rank #{cost_pos} of {len(cost_rank)}. "
                        f"Capacity fully allocated to lower-cost vendors meeting SLA."
                    )
            else:
                if vendor_id == cost_rank[0]:
                    reason = (
                        f"Allocated {allocation:.1%} — cheapest vendor "
                        f"(${vendor.cost_per_unit}/unit) meeting SLA and reliability requirements."
                    )
                elif vendor.lead_time_days <= self.sla_threshold_days:
                    reason = (
                        f"Allocated {allocation:.1%} — meets SLA ({vendor.lead_time_days}d ≤ "
                        f"{self.sla_threshold_days}d threshold) with {vendor.reliability:.0%} reliability. "
                        f"Used to fulfill remaining demand after cheaper vendors hit capacity."
                    )
                else:
                    reason = (
                        f"Allocated {allocation:.1%} despite SLA breach — "
                        f"no better alternative available within budget constraints."
                    )

            explanations.append(VendorExplanation(
                vendor_id=vendor_id,
                allocation=allocation,
                cost_contribution=cost_contribution,
                sla_status=sla_status,
                reliability_risk=reliability_risk,
                inclusion_reason=reason,
            ))

        return explanations

    def _explain_objective(
        self,
        solution: Dict[str, float],
        demand: float,
    ) -> ObjectiveExplanation:
        """Decompose objective into weighted components."""
        x = np.array([solution.get(v.vendor_id, 0.0) for v in self.vendors])
        costs = np.array([v.cost_per_unit for v in self.vendors])
        lead_times = np.array([v.lead_time_days for v in self.vendors])
        reliabilities = np.array([v.reliability for v in self.vendors])

        cost_term = float(np.dot(x, costs) * demand)
        overdue = np.maximum(0.0, lead_times - self.sla_threshold_days)
        sla_term = float(np.dot(x, overdue) * self.penalty_per_day * demand)
        risk_term = float(np.dot(x, 1.0 - reliabilities))

        weighted_cost = self.w_cost * cost_term
        weighted_sla = self.w_sla * sla_term
        weighted_risk = self.w_risk * risk_term
        weighted_total = weighted_cost + weighted_sla + weighted_risk

        # Contribution percentages
        if weighted_total > 0:
            cost_pct = weighted_cost / weighted_total * 100
            sla_pct = weighted_sla / weighted_total * 100
            risk_pct = weighted_risk / weighted_total * 100
        else:
            cost_pct = sla_pct = risk_pct = 0.0

        # Dominant term
        contributions = {
            "cost": weighted_cost,
            "sla_penalty": weighted_sla,
            "reliability_risk": weighted_risk,
        }
        dominant_term = max(contributions, key=contributions.get)

        return ObjectiveExplanation(
            weighted_total=weighted_total,
            cost_term=cost_term,
            cost_weight=self.w_cost,
            cost_contribution_pct=cost_pct,
            sla_term=sla_term,
            sla_weight=self.w_sla,
            sla_contribution_pct=sla_pct,
            risk_term=risk_term,
            risk_weight=self.w_risk,
            risk_contribution_pct=risk_pct,
            dominant_term=dominant_term,
        )

    def _explain_constraints(
        self,
        solution: Dict[str, float],
        demand: float,
    ) -> List[ConstraintExplanation]:
        """Evaluate and explain each constraint's status."""
        explanations = []

        # Budget constraint
        total_cost = sum(
            solution.get(v.vendor_id, 0.0) * v.cost_per_unit * demand
            for v in self.vendors
        )
        budget_slack = self.budget_cap - total_cost
        budget_status = "satisfied" if budget_slack >= 0 else "violated"
        if abs(budget_slack) < self.budget_cap * 0.02:
            budget_status = "active"   # within 2% of boundary

        explanations.append(ConstraintExplanation(
            name="budget_cap",
            severity="hard",
            status=budget_status,
            slack=budget_slack,
            narrative=(
                f"Total cost ${total_cost:,.0f} vs. budget cap ${self.budget_cap:,.0f}. "
                f"${budget_slack:,.0f} headroom remaining."
                if budget_slack >= 0 else
                f"Budget violated by ${-budget_slack:,.0f}. Infeasible solution."
            ),
        ))

        # SLA constraint
        weighted_lead = sum(
            solution.get(v.vendor_id, 0.0) * v.lead_time_days
            for v in self.vendors
        )
        sla_slack = self.sla_threshold_days - weighted_lead
        sla_status = "satisfied" if sla_slack >= 0 else "violated"

        explanations.append(ConstraintExplanation(
            name="sla_threshold",
            severity="soft",
            status=sla_status,
            slack=sla_slack,
            narrative=(
                f"Weighted average lead time {weighted_lead:.2f}d vs. "
                f"SLA threshold {self.sla_threshold_days}d. "
                f"{'Within SLA — no penalty.' if sla_slack >= 0 else f'Exceeds SLA by {-sla_slack:.2f}d — penalty applies.'}"
            ),
        ))

        # Allocation sum constraint
        alloc_sum = sum(solution.get(v.vendor_id, 0.0) for v in self.vendors)
        alloc_slack = alloc_sum - 1.0
        alloc_status = "satisfied" if abs(alloc_slack) < 1e-4 else "violated"

        explanations.append(ConstraintExplanation(
            name="allocation_sum",
            severity="hard",
            status=alloc_status,
            slack=-abs(alloc_slack),
            narrative=(
                f"Allocation sums to {alloc_sum:.6f} "
                f"({'✓ valid' if alloc_status == 'satisfied' else '✗ invalid — solver error'})."
            ),
        ))

        return explanations

    def _explain_sensitivity(
        self,
        solution: Dict[str, float],
        demand: float,
    ) -> SensitivityExplanation:
        """
        Compute first-order sensitivity of objective to weight changes.

        Math:
            objective = w_cost * cost + w_sla * sla + w_risk * risk
            ∂obj/∂w_cost = cost  (at current solution)
            ∂obj/∂w_sla  = sla
            ∂obj/∂w_risk = risk

        Interpretation:
            If ∂obj/∂w_cost = 39,600, then increasing w_cost by 0.01
            increases the objective by ~396 — pushing solver toward
            cheaper (but potentially riskier) allocations in future runs.
        """
        x = np.array([solution.get(v.vendor_id, 0.0) for v in self.vendors])
        costs = np.array([v.cost_per_unit for v in self.vendors])
        lead_times = np.array([v.lead_time_days for v in self.vendors])
        reliabilities = np.array([v.reliability for v in self.vendors])

        cost_term = float(np.dot(x, costs) * demand)
        overdue = np.maximum(0.0, lead_times - self.sla_threshold_days)
        sla_term = float(np.dot(x, overdue) * self.penalty_per_day * demand)
        risk_term = float(np.dot(x, 1.0 - reliabilities))

        sensitivities = {
            "w_cost": abs(cost_term),
            "w_sla": abs(sla_term),
            "w_risk": abs(risk_term),
        }
        most_sensitive = max(sensitivities, key=sensitivities.get)

        narrative = (
            f"The objective is most sensitive to changes in {most_sensitive}. "
            f"A 0.01 increase in w_cost shifts the objective by ~{cost_term * 0.01:,.0f}. "
            f"SLA weight sensitivity is {'high' if sla_term > cost_term * 0.1 else 'low'} "
            f"because {'vendors are breaching SLA' if sla_term > 0 else 'all vendors meet SLA — SLA penalty is zero'}."
        )

        return SensitivityExplanation(
            d_objective_d_w_cost=cost_term,
            d_objective_d_w_sla=sla_term,
            d_objective_d_w_risk=risk_term,
            most_sensitive_weight=most_sensitive,
            narrative=narrative,
        )

    def _build_summary(
        self,
        solution: Dict[str, float],
        demand: float,
        total_cost: float,
        objective: ObjectiveExplanation,
        constraints: List[ConstraintExplanation],
    ) -> str:
        """Build one-paragraph plain-English summary for non-technical stakeholders."""
        active_vendors = [
            vid for vid, frac in solution.items() if frac >= 1e-4
        ]
        hard_violations = [c for c in constraints if c.status == "violated" and c.severity == "hard"]
        sla_ok = all(c.status == "satisfied" for c in constraints if c.name == "sla_threshold")

        if hard_violations:
            return (
                f"Decision is INFEASIBLE. Hard constraint violations: "
                f"{', '.join(c.name for c in hard_violations)}. "
                f"No valid allocation exists under current constraints."
            )

        vendor_str = ", ".join(
            f"{vid} ({solution[vid]:.0%})" for vid in active_vendors
        )
        sla_str = "within SLA" if sla_ok else "with SLA penalty"

        return (
            f"Allocated {demand:,.0f} units across {len(active_vendors)} vendor(s): {vendor_str}. "
            f"Total cost ${total_cost:,.0f} {sla_str}. "
            f"Objective dominated by {objective.dominant_term} term "
            f"({objective.cost_contribution_pct:.0f}% cost, "
            f"{objective.sla_contribution_pct:.0f}% SLA, "
            f"{objective.risk_contribution_pct:.0f}% risk). "
            f"All hard constraints satisfied."
        )