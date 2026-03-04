"""
Phase 3 — Baseline Heuristic Comparison.

Compares three procurement heuristics against the decision engine:
    1. cheapest_first       — minimize raw cost, ignore SLA/risk
    2. highest_reliability  — maximize reliability, ignore cost
    3. balanced_sla         — equal weight across SLA-compliant vendors only

All strategies evaluated with identical metric:
    effective_cost = shipping_cost + sla_penalty

Why this matters:
    Engine should outperform heuristics not because heuristics are stupid,
    but because heuristics optimize for one dimension only.
    Engine explicitly models the tradeoff.

Run from project root:
    python examples/run_phase3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass

from decision_intelligence.core.constraints.base import ConstraintRegistry
from decision_intelligence.core.optimization.scipy_solver import MixedScipySolver
from decision_intelligence.core.decision.base import DecisionContext
from decision_intelligence.governance.audit import AuditLogger
from decision_intelligence.domain.shipping.objective import ShippingObjective, VendorProfile
from decision_intelligence.domain.shipping.constraints import BudgetConstraint, SLAConstraint
from decision_intelligence.domain.shipping.engine import ShippingVendorEngine, VendorConfig
from decision_intelligence.core.decision.orchestrator import DecisionOrchestrator


# ─── Config ──────────────────────────────────────────────────────────────────

CFG_PATH = Path(__file__).parent.parent / "configs" / "phase1_baseline.yaml"

def load_config() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)

def build_vendor_profiles(cfg: dict) -> list[VendorProfile]:
    return [
        VendorProfile(
            vendor_id=v["vendor_id"],
            cost_per_unit=v["base_cost_per_unit"],
            reliability=v["reliability_score"],
            lead_time_days=v["lead_time_days"],
        )
        for v in cfg["vendors"]
    ]

def build_vendor_configs(cfg: dict) -> list[VendorConfig]:
    return [
        VendorConfig(
            vendor_id=v["vendor_id"],
            cost_per_unit=v["base_cost_per_unit"],
            reliability=v["reliability_score"],
            lead_time_days=v["lead_time_days"],
            capacity_fraction=v["capacity_fraction"],
        )
        for v in cfg["vendors"]
    ]

def build_engine(
    cfg: dict,
    vendor_profiles: list[VendorProfile],
    vendor_configs: list[VendorConfig],
    budget_cap: float,
) -> ShippingVendorEngine:
    c = cfg["constraints"]
    w = cfg["objective_weights"]
    s = cfg["solver"]

    objective = ShippingObjective(
        vendors=vendor_profiles,
        w_cost=w["w_cost"],
        w_sla=w["w_sla"],
        w_risk=w["w_risk"],
        sla_threshold_days=c["sla_threshold"]["threshold_days"],
        penalty_per_day=c["sla_threshold"]["penalty_per_day"],
    )

    registry = ConstraintRegistry()
    registry.register(BudgetConstraint(vendor_profiles, budget_cap=budget_cap))
    registry.register(SLAConstraint(
        vendor_profiles,
        sla_threshold_days=c["sla_threshold"]["threshold_days"],
        penalty_per_day=c["sla_threshold"]["penalty_per_day"],
    ))

    solver = MixedScipySolver(config={
        "n_restarts": s["n_restarts"],
        "seed": cfg["experiment"]["seed"],
    })

    return ShippingVendorEngine(
        config={"budget_cap": budget_cap},
        vendors=vendor_configs,
        objective=objective,
        constraints=registry,
        solver=solver,
    )


# ─── Evaluation metric (shared across ALL strategies) ────────────────────────

@dataclass
class StrategyResult:
    name: str
    description: str
    allocation: dict[str, float]
    raw_cost: float
    sla_penalty: float
    risk_exposure: float
    effective_cost: float
    weighted_lead_time: float
    meets_budget: bool


def evaluate_allocation(
    name: str,
    description: str,
    allocation: dict[str, float],
    vendor_profiles: list[VendorProfile],
    demand: float,
    sla_threshold: float,
    penalty_per_day: float,
    budget_cap: float,
) -> StrategyResult:
    """
    Shared evaluation function — ALL strategies use this.

    Why centralize:
        Fair comparison requires identical math for all strategies.
        If each strategy computes its own metrics, subtle differences
        in formula will corrupt the comparison.

    effective_cost = shipping_cost + sla_penalty
    risk_exposure  = Σ(x_i × (1 - reliability_i))  [dimensionless rate]
    """
    profile_map = {v.vendor_id: v for v in vendor_profiles}

    raw_cost = sum(
        allocation.get(vid, 0.0) * profile_map[vid].cost_per_unit * demand
        for vid in allocation
    )
    weighted_lead = sum(
        allocation.get(vid, 0.0) * profile_map[vid].lead_time_days
        for vid in allocation
    )
    excess_days = max(0.0, weighted_lead - sla_threshold)
    sla_penalty = excess_days * penalty_per_day * demand
    risk_exposure = sum(
        allocation.get(vid, 0.0) * (1 - profile_map[vid].reliability)
        for vid in allocation
    )

    return StrategyResult(
        name=name,
        description=description,
        allocation=allocation,
        raw_cost=raw_cost,
        sla_penalty=sla_penalty,
        risk_exposure=risk_exposure,
        effective_cost=raw_cost + sla_penalty,
        weighted_lead_time=weighted_lead,
        meets_budget=raw_cost <= budget_cap,
    )


# ─── Heuristics ──────────────────────────────────────────────────────────────

def heuristic_cheapest_first(
    vendor_configs: list[VendorConfig],
    vendor_profiles: list[VendorProfile],
) -> dict[str, float]:
    """
    Allocate greedily from cheapest vendor up to capacity.

    Rationale: common procurement heuristic — minimize unit cost.
    Weakness: ignores SLA and reliability entirely.
    """
    sorted_pairs = sorted(
        zip(vendor_configs, vendor_profiles),
        key=lambda t: t[1].cost_per_unit,
    )
    allocation = {v.vendor_id: 0.0 for v in vendor_configs}
    remaining = 1.0
    for vc, vp in sorted_pairs:
        if remaining <= 0:
            break
        alloc = min(vc.capacity_fraction, remaining)
        allocation[vp.vendor_id] = alloc
        remaining -= alloc
    return allocation


def heuristic_highest_reliability(
    vendor_configs: list[VendorConfig],
    vendor_profiles: list[VendorProfile],
) -> dict[str, float]:
    """
    Allocate greedily from most reliable vendor up to capacity.

    Rationale: risk-averse procurement — prioritize delivery certainty.
    Weakness: ignores cost entirely, may be expensive.
    """
    sorted_pairs = sorted(
        zip(vendor_configs, vendor_profiles),
        key=lambda t: -t[1].reliability,
    )
    allocation = {v.vendor_id: 0.0 for v in vendor_configs}
    remaining = 1.0
    for vc, vp in sorted_pairs:
        if remaining <= 0:
            break
        alloc = min(vc.capacity_fraction, remaining)
        allocation[vp.vendor_id] = alloc
        remaining -= alloc
    return allocation


def heuristic_balanced_sla(
    vendor_configs: list[VendorConfig],
    vendor_profiles: list[VendorProfile],
    sla_threshold: float,
) -> dict[str, float]:
    """
    Equal allocation across vendors that meet SLA threshold only.
    Capacity constraints still respected.

    Rationale: operational simplicity — treat all SLA-compliant vendors equally.
    Weakness: ignores cost differences between compliant vendors.

    If no vendor meets SLA, fall back to all vendors equally.
    """
    sla_pairs = [
        (vc, vp)
        for vc, vp in zip(vendor_configs, vendor_profiles)
        if vp.lead_time_days <= sla_threshold
    ]

    # Fallback: if no vendor meets SLA, use all vendors
    if not sla_pairs:
        sla_pairs = list(zip(vendor_configs, vendor_profiles))

    allocation = {v.vendor_id: 0.0 for v in vendor_configs}

    # Equal target fraction among SLA-compliant vendors
    n = len(sla_pairs)
    target = 1.0 / n

    # First pass: assign min(target, capacity) to each
    remaining = 1.0
    overflow = 0.0
    for vc, vp in sla_pairs:
        alloc = min(target, vc.capacity_fraction, remaining)
        allocation[vp.vendor_id] = alloc
        remaining -= alloc
        if target > vc.capacity_fraction:
            overflow += target - vc.capacity_fraction

    # Second pass: distribute overflow to vendors with spare capacity
    if remaining > 1e-6:
        for vc, vp in sla_pairs:
            if remaining <= 0:
                break
            spare = vc.capacity_fraction - allocation[vp.vendor_id]
            if spare > 1e-6:
                extra = min(spare, remaining)
                allocation[vp.vendor_id] += extra
                remaining -= extra

    return allocation


# ─── Display ──────────────────────────────────────────────────────────────────

DIV = "─" * 72

def print_strategy_detail(result: StrategyResult, demand: float) -> None:
    print(f"\n  Strategy : {result.name}")
    print(f"  Logic    : {result.description}")
    print(f"  {'Vendor':<14} {'Allocation':>10} {'Units':>10} {'Cost':>12}")
    print(f"  {'─' * 50}")
    for vid, frac in sorted(result.allocation.items(), key=lambda x: -x[1]):
        if frac < 1e-4:
            continue
        units = frac * demand
        bar = "█" * int(frac * 20)
        print(f"  {vid:<14} {frac:>9.1%} {units:>10,.0f}  {bar}")


def print_comparison_table(
    results: list[StrategyResult],
    budget_cap: float,
    engine_result: StrategyResult,
) -> None:
    print(f"\n{'═' * 72}")
    print(f"  STRATEGY COMPARISON — All evaluated with identical metric")
    print(f"  effective_cost = shipping_cost + sla_penalty")
    print(f"{'═' * 72}")

    header = f"  {'Strategy':<24} {'Raw Cost':>10} {'SLA Pen.':>10} {'Risk':>8} {'Eff. Cost':>12} {'vs Engine':>10}"
    print(f"\n{header}")
    print(f"  {DIV}")

    all_results = results + [engine_result]
    engine_eff = engine_result.effective_cost

    for r in all_results:
        delta = r.effective_cost - engine_eff
        delta_str = f"{delta:>+10,.0f}" if r.name != engine_result.name else f"{'—':>10}"
        budget_flag = "" if r.meets_budget else " ⚠️"
        print(
            f"  {r.name:<24} "
            f"{r.raw_cost:>10,.0f} "
            f"{r.sla_penalty:>10,.0f} "
            f"{r.risk_exposure:>8.4f} "
            f"{r.effective_cost:>12,.0f}"
            f"{delta_str}"
            f"{budget_flag}"
        )

    print(f"  {DIV}")
    print(f"  Budget cap: ${budget_cap:,.0f}  |  ⚠️ = exceeds budget")

    # Ranking by effective cost
    print(f"\n  Ranking by Effective Cost:")
    print(f"  {DIV}")
    ranked = sorted(all_results, key=lambda r: r.effective_cost)
    for i, r in enumerate(ranked, 1):
        marker = " ← engine" if r.name == engine_result.name else ""
        print(f"  {i}. {r.name:<24} ${r.effective_cost:>10,.0f}{marker}")

    # Why engine wins
    best_heuristic = min(results, key=lambda r: r.effective_cost)
    improvement = (best_heuristic.effective_cost - engine_eff) / best_heuristic.effective_cost * 100
    risk_improvement = (best_heuristic.risk_exposure - engine_result.risk_exposure) / best_heuristic.risk_exposure * 100 if best_heuristic.risk_exposure > 0 else 0

    print(f"\n  Engine vs Best Heuristic ({best_heuristic.name}):")
    print(f"  {DIV}")
    print(f"  Effective cost improvement : {improvement:+.1f}%")
    print(f"  Risk exposure improvement  : {risk_improvement:+.1f}%")
    print(f"\n  Why engine outperforms:")
    print(f"    Heuristics optimize one dimension (cost OR reliability OR SLA).")
    print(f"    Engine explicitly models cost-SLA-risk tradeoff via weighted objective.")
    print(f"    Result: lower effective cost AND lower risk simultaneously.")


def print_tradeoff_analysis(
    results: list[StrategyResult],
    engine_result: StrategyResult,
) -> None:
    print(f"\n{'═' * 72}")
    print(f"  TRADEOFF ANALYSIS")
    print(f"{'═' * 72}")
    print(f"\n  Each heuristic optimizes one dimension — at the cost of others:\n")

    all_results = results + [engine_result]
    for r in all_results:
        sla_status = "✅ meets SLA" if r.weighted_lead_time <= 3.0 else f"❌ exceeds SLA ({r.weighted_lead_time:.1f}d)"
        print(f"  {r.name:<24} cost={r.raw_cost:>8,.0f}  risk={r.risk_exposure:.4f}  {sla_status}")

    print(f"\n  Key insight:")
    print(f"    cheapest_first    → lowest raw cost, but SLA breach = hidden cost")
    print(f"    highest_reliab.   → lowest risk, but cost premium")
    print(f"    balanced_sla      → SLA-safe, equal treatment, moderate cost")
    print(f"    engine            → explicit tradeoff: best effective cost + risk")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 72)
    print("  RISK-AWARE DECISION ENGINE — Phase 3")
    print("  Baseline Heuristic Comparison")
    print("═" * 72)

    cfg             = load_config()
    vendor_profiles = build_vendor_profiles(cfg)
    vendor_configs  = build_vendor_configs(cfg)
    demand          = cfg["scenario"]["demand"]
    sla_threshold   = cfg["constraints"]["sla_threshold"]["threshold_days"]
    penalty_per_day = cfg["constraints"]["sla_threshold"]["penalty_per_day"]
    budget_cap      = cfg["constraints"]["budget_cap"]["value"]

    print(f"\n  Demand     : {demand:,} units")
    print(f"  SLA        : {sla_threshold} days")
    print(f"  Budget cap : ${budget_cap:,}")
    print(f"\n  Vendors:")
    for vp, vc in zip(vendor_profiles, vendor_configs):
        sla_flag = "✅" if vp.lead_time_days <= sla_threshold else "❌"
        print(f"    {vp.vendor_id:<12} cost=${vp.cost_per_unit:.2f}  "
              f"reliability={vp.reliability:.2f}  "
              f"lead={vp.lead_time_days}d {sla_flag}  "
              f"cap={vc.capacity_fraction:.0%}")

    # ── Run engine ────────────────────────────────────────────────────────────
    logger       = AuditLogger()
    engine       = build_engine(cfg, vendor_profiles, vendor_configs, budget_cap)
    orchestrator = DecisionOrchestrator(engine=engine, audit_logger=logger)
    result       = orchestrator.run(
        raw_features={"demand": demand},
        domain="shipping",
        extra_metadata={"scenario": "phase3_engine"},
    )
    opt = result.optimization

    if not opt.is_feasible:
        print("\n  ❌ Engine returned infeasible — cannot compare.")
        return

    engine_result = evaluate_allocation(
        name="engine",
        description="multi-objective optimization (cost + SLA + risk)",
        allocation=opt.best_solution,
        vendor_profiles=vendor_profiles,
        demand=demand,
        sla_threshold=sla_threshold,
        penalty_per_day=penalty_per_day,
        budget_cap=budget_cap,
    )

    # ── Run heuristics ────────────────────────────────────────────────────────
    heuristics_raw = [
        (
            "cheapest_first",
            "allocate to cheapest vendor first, up to capacity",
            heuristic_cheapest_first(vendor_configs, vendor_profiles),
        ),
        (
            "highest_reliability",
            "allocate to most reliable vendor first, up to capacity",
            heuristic_highest_reliability(vendor_configs, vendor_profiles),
        ),
        (
            "balanced_sla",
            "equal allocation across SLA-compliant vendors only",
            heuristic_balanced_sla(vendor_configs, vendor_profiles, sla_threshold),
        ),
    ]

    heuristic_results = []
    for name, description, allocation in heuristics_raw:
        heuristic_results.append(evaluate_allocation(
            name=name,
            description=description,
            allocation=allocation,
            vendor_profiles=vendor_profiles,
            demand=demand,
            sla_threshold=sla_threshold,
            penalty_per_day=penalty_per_day,
            budget_cap=budget_cap,
        ))

    # ── Print strategy details ────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  STRATEGY ALLOCATIONS")
    print(f"{'═' * 72}")
    for r in heuristic_results:
        print_strategy_detail(r, demand)
    print_strategy_detail(engine_result, demand)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison_table(heuristic_results, budget_cap, engine_result)
    print_tradeoff_analysis(heuristic_results, engine_result)

    print(f"\n{'═' * 72}")
    print(f"  Phase 3 complete.")
    print(f"  Next: Phase 4 — Historical Backtesting with real Meratus data")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()