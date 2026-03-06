"""
Phase 2 — Risk Quantification Under Demand Uncertainty.

Pipeline:
    1. Run deterministic optimizer (identical to Phase 1)
    2. Take optimal allocation as fixed
    3. Stress-test allocation via Monte Carlo (10,000 demand scenarios)
    4. Report: Expected cost, CVaR-95, P(budget exceed), P(SLA breach)

Why this order:
    Optimizer finds the best allocation under deterministic assumptions.
    Monte Carlo evaluates how robust that allocation is under uncertainty.
    These are separate questions — separation keeps each component testable.

Run from project root:
    python examples/run_phase2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
from datetime import datetime, timezone

from decision_intelligence.core.constraints.base import ConstraintRegistry
from decision_intelligence.core.optimization.scipy_solver import MixedScipySolver
from decision_intelligence.core.decision.base import DecisionContext
from decision_intelligence.core.optimization.base import OptimizationResult
from decision_intelligence.governance.audit import AuditLogger
from decision_intelligence.domain.shipping.objective import ShippingObjective, VendorProfile
from decision_intelligence.domain.shipping.constraints import BudgetConstraint, SLAConstraint
from decision_intelligence.domain.shipping.engine import ShippingVendorEngine, VendorConfig
from decision_intelligence.domain.shipping.risk import ShippingRiskEvaluator
from decision_intelligence.core.risk.monte_carlo import DemandDistribution
from decision_intelligence.core.decision.orchestrator import DecisionOrchestrator


#  Config 

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


#  Display 

DIV = "─" * 65


def print_optimizer_result(
    opt: OptimizationResult,
    demand: float,
    vendor_profiles: list[VendorProfile],
) -> None:
    print(f"\n{'═' * 65}")
    print(f"  STEP 1: DETERMINISTIC OPTIMIZER")
    print(f"{'═' * 65}")

    if not opt.is_feasible:
        print(f"\n  ❌  INFEASIBLE — cannot proceed to risk evaluation")
        return

    print(f"\n  ✅  Optimal allocation found (demand = {demand:,.0f} units)\n")

    solution = opt.best_solution
    profile_map = {v.vendor_id: v for v in vendor_profiles}

    print(f"  {'Vendor':<14} {'Allocation':>10} {'Units':>10} {'Cost':>12}")
    print(f"  {DIV}")

    total_cost = 0.0
    for vid, frac in sorted(solution.items(), key=lambda x: -x[1]):
        if frac < 1e-4:
            continue
        vp = profile_map[vid]
        units = frac * demand
        cost = units * vp.cost_per_unit
        total_cost += cost
        bar = "█" * int(frac * 20)
        print(f"  {vid:<14} {frac:>9.1%} {units:>10,.0f} {cost:>12,.0f}  {bar}")

    print(f"  {DIV}")
    print(f"  {'TOTAL':<14} {'100.0%':>10} {demand:>10,.0f} ${total_cost:>11,.0f}")
    print(f"\n  Solver: SLSQP | Iterations: {opt.iterations} | Converged: {opt.converged}")


def print_risk_report(risk_report, budget_cap: float, n_simulations: int) -> None:
    print(f"\n{'═' * 65}")
    print(f"  STEP 2: MONTE CARLO RISK EVALUATION")
    print(f"  {n_simulations:,} demand scenarios — "
          f"Normal(mu={risk_report.extra_metrics['demand_mu']:,.0f}, "
          f"sigma={risk_report.extra_metrics['demand_sigma']:,.0f}), truncated at 0")
    print(f"{'═' * 65}")

    m = risk_report.extra_metrics
    expected_cost    = m["expected_cost"]
    cost_std         = m["cost_std"]
    p_budget         = m["p_budget_exceed"]
    p_sla            = m["p_sla_breach"]
    var_95           = risk_report.var_95
    cvar_95          = risk_report.cvar_95

    print(f"\n  Cost Distribution Across {n_simulations:,} Scenarios:")
    print(f"  {DIV}")
    print(f"  {'Expected cost':<35} ${expected_cost:>12,.0f}")
    print(f"  {'Cost std dev':<35} ${cost_std:>12,.0f}")
    print(f"  {'VaR-95 (95th percentile)':<35} ${var_95:>12,.0f}")
    print(f"  {'CVaR-95 (mean of worst 5%)':<35} ${cvar_95:>12,.0f}")
    print(f"  {'Budget cap':<35} ${budget_cap:>12,.0f}")

    print(f"\n  Risk Flags:")
    print(f"  {DIV}")

    # P(budget exceed)
    p_budget_pct = p_budget * 100
    budget_icon = "✅" if p_budget <= 0.05 else ("⚠️ " if p_budget <= 0.10 else "❌")
    print(f"  {budget_icon} P(cost > budget_cap)     = {p_budget_pct:>6.1f}%", end="")
    if p_budget <= 0.05:
        print(f"  — low risk")
    elif p_budget <= 0.10:
        print(f"  — moderate risk, monitor")
    else:
        print(f"  — HIGH RISK, action required")

    # P(SLA breach)
    p_sla_pct = p_sla * 100
    sla_icon = "✅" if p_sla <= 0.05 else ("⚠️ " if p_sla <= 0.15 else "❌")
    print(f"  {sla_icon} P(SLA breach)            = {p_sla_pct:>6.1f}%", end="")
    if p_sla == 0.0:
        print(f"  — allocation always meets SLA under this distribution")
    elif p_sla <= 0.05:
        print(f"  — low risk")
    elif p_sla <= 0.15:
        print(f"  — moderate risk")
    else:
        print(f"  — HIGH RISK")

    # Overall tolerance
    print(f"\n  Overall Risk Tolerance: ", end="")
    if risk_report.is_within_tolerance:
        print(f" WITHIN TOLERANCE")
    else:
        print(f" EXCEEDS TOLERANCE")

    print(f"\n  Interpretation:")
    print(f"  {DIV}")
    print(f"  Under demand uncertainty (CV=15%), the optimal allocation")
    print(f"  from Phase 1 produces an expected cost of ${expected_cost:,.0f}.")
    print(f"  In the worst 5% of scenarios (CVaR-95), cost reaches ${cvar_95:,.0f}.")
    cvar_vs_budget = (cvar_95 / budget_cap - 1) * 100
    if cvar_vs_budget > 0:
        print(f"  CVaR-95 exceeds budget cap by {cvar_vs_budget:.1f}% —")
        print(f"  consider tightening allocation or increasing budget buffer.")
    else:
        print(f"  CVaR-95 is {-cvar_vs_budget:.1f}% below budget cap —")
        print(f"  allocation is robust even in worst-case demand scenarios.")


def print_phase_comparison(
    deterministic_cost: float,
    expected_cost: float,
    cvar_95: float,
    budget_cap: float,
) -> None:
    print(f"\n{'═' * 65}")
    print(f"  PHASE 1 vs PHASE 2 — What Uncertainty Adds")
    print(f"{'═' * 65}")
    print(f"\n  {'Metric':<40} {'Value':>12}")
    print(f"  {DIV}")
    print(f"  {'Phase 1 — Deterministic cost':<40} ${deterministic_cost:>11,.0f}")
    print(f"  {'Phase 2 — Expected cost (E[cost])':<40} ${expected_cost:>11,.0f}")
    print(f"  {'Phase 2 — CVaR-95 (worst 5% avg)':<40} ${cvar_95:>11,.0f}")
    print(f"  {'Budget cap':<40} ${budget_cap:>11,.0f}")
    print(f"  {DIV}")

    uplift = (expected_cost - deterministic_cost) / deterministic_cost * 100
    cvar_uplift = (cvar_95 - deterministic_cost) / deterministic_cost * 100

    print(f"\n  Expected cost uplift vs deterministic : {uplift:+.1f}%")
    print(f"  CVaR-95 uplift vs deterministic       : {cvar_uplift:+.1f}%")
    print(f"\n  Why this matters:")
    print(f"    Phase 1 optimizer assumed demand = {deterministic_cost/3.96:.0f} units exactly.")
    print(f"    Phase 2 shows that under realistic demand variance,")
    print(f"    expected cost rises by {uplift:+.1f}% and worst-case by {cvar_uplift:+.1f}%.")
    print(f"    A system that ignores this is not risk-aware — it is optimistic.")


#   Main 

def main() -> None:
    print("\n" + "═" * 65)
    print("  RISK-AWARE DECISION ENGINE — Phase 2")
    print("  Uncertainty Quantification via Monte Carlo")
    print("═" * 65)

    cfg             = load_config()
    vendor_profiles = build_vendor_profiles(cfg)
    vendor_configs  = build_vendor_configs(cfg)
    demand          = cfg["scenario"]["demand"]
    sla_threshold   = cfg["constraints"]["sla_threshold"]["threshold_days"]
    penalty_per_day = cfg["constraints"]["sla_threshold"]["penalty_per_day"]
    budget_cap      = cfg["constraints"]["budget_cap"]["value"]
    seed            = cfg["experiment"]["seed"]
    n_simulations   = 10_000

    print(f"\n  Experiment    : {cfg['experiment']['name']} (Phase 2)")
    print(f"  Seed          : {seed}  (deterministic)")
    print(f"  Vendors       : {len(vendor_profiles)}")
    print(f"  Demand (det.) : {demand:,} units")
    print(f"  Budget cap    : ${budget_cap:,}")
    print(f"  Simulations   : {n_simulations:,}")
    print(f"  Demand dist.  : Normal(mu={demand:,}, sigma=1500), truncated at 0")

    logger = AuditLogger()

    # Step 1: Deterministic optimizer (Phase 1 pipeline, unchanged) ─────────
    engine       = build_engine(cfg, vendor_profiles, vendor_configs, budget_cap)
    orchestrator = DecisionOrchestrator(engine=engine, audit_logger=logger)

    result = orchestrator.run(
        raw_features={"demand": demand},
        domain="shipping",
        extra_metadata={"scenario": "phase2_baseline"},
    )

    opt = result.optimization
    print_optimizer_result(opt, demand, vendor_profiles)

    if not opt.is_feasible:
        print("\n  Cannot proceed to risk evaluation — optimizer returned infeasible.")
        return

    #  Step 2: Monte Carlo risk evaluation 
    distribution = DemandDistribution(
        mu=float(demand),
        sigma=1500.0,
        lower=0.0,
    )

    risk_evaluator = ShippingRiskEvaluator(
        config={
            "n_simulations": n_simulations,
            "seed": seed,
            "budget_cap": budget_cap,
            "cvar_alpha": 0.05,
            "max_p_budget_exceed": 0.10,
        },
        vendors=vendor_profiles,
        demand_distribution=distribution,
        sla_threshold_days=sla_threshold,
        penalty_per_day=penalty_per_day,
    )

    risk_report = risk_evaluator.assess(
        solution=opt.best_solution,
        context={"features": {"demand": demand}},
    )

    print_risk_report(risk_report, budget_cap, n_simulations)

    #  Step 3: Phase 1 vs Phase 2 comparison 
    x = np.array([
        opt.best_solution.get(v.vendor_id, 0.0)
        for v in vendor_profiles
    ])
    costs_arr = np.array([v.cost_per_unit for v in vendor_profiles])
    deterministic_cost = float(np.dot(x, costs_arr) * demand)

    print_phase_comparison(
        deterministic_cost=deterministic_cost,
        expected_cost=risk_report.extra_metrics["expected_cost"],
        cvar_95=risk_report.cvar_95,
        budget_cap=budget_cap,
    )

    print(f"\n{'═' * 65}")
    print(f"  Phase 2 complete.")
    print(f"  Audit log: {len(logger.records())} records captured.")
    print(f"  Next: Phase 3 — Backtesting + Decision Regret Metric")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()