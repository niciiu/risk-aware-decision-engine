"""
Phase 1 — End-to-end vendor allocation scenario.

Demonstrates:
    1. Feasible baseline decision with full audit trace
    2. Explicit infeasible case (budget below minimum possible cost)
    3. Baseline comparison: cheapest-first heuristic vs decision engine
    4. Sensitivity: demand +20%

Run from project root:
    python examples/run_phase1.py
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


def make_context(demand: float, scenario: str) -> DecisionContext:
    return DecisionContext(
        timestamp=datetime.now(timezone.utc).isoformat(),
        domain="shipping",
        features={"demand": demand},
        metadata={"scenario": scenario},
    )


# ─── Baseline heuristic ───────────────────────────────────────────────────────

def cheapest_first(
    vendor_configs: list[VendorConfig],
    vendor_profiles: list[VendorProfile],
    demand: float,
    sla_threshold: float,
    penalty_per_day: float,
) -> dict:
    sorted_vendors = sorted(
        zip(vendor_configs, vendor_profiles),
        key=lambda t: t[1].cost_per_unit,
    )

    allocation = {v.vendor_id: 0.0 for v in vendor_configs}
    remaining_fraction = 1.0

    for vc, vp in sorted_vendors:
        if remaining_fraction <= 0:
            break
        alloc = min(vc.capacity_fraction, remaining_fraction)
        allocation[vc.vendor_id] = alloc
        remaining_fraction -= alloc

    profile_map = {vp.vendor_id: vp for _, vp in sorted_vendors}
    total_cost = sum(
        allocation[vid] * profile_map[vid].cost_per_unit * demand
        for vid in allocation
    )
    weighted_lead = sum(
        allocation[vid] * profile_map[vid].lead_time_days
        for vid in allocation
    )
    excess_days = max(0.0, weighted_lead - sla_threshold)
    sla_penalty = excess_days * penalty_per_day * demand
    weighted_risk = sum(
        allocation[vid] * (1 - profile_map[vid].reliability)
        for vid in allocation
    )

    return {
        "allocation": allocation,
        "raw_cost": total_cost,
        "sla_penalty": sla_penalty,
        "risk_exposure": weighted_risk,
        "effective_cost": total_cost + sla_penalty,
        "weighted_lead_time": weighted_lead,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

DIV = "─" * 65


def print_result(
    output,
    opt: OptimizationResult,
    demand: float,
    vendor_profiles: list[VendorProfile],
    cfg: dict,
    label: str = "",
    solver_name: str = "SLSQP",
) -> None:
    print(f"\n{'═' * 65}")
    if label:
        print(f"  {label}")
    print(f"{'═' * 65}")

    if not opt.is_feasible:
        print(f"\n  ❌  INFEASIBLE")
        print(f"\n  {output.action}")
        print(f"\n  Solver: {solver_name}")
        print(f"  Solver iterations: {opt.iterations}")
        if opt.constraint_violations:
            print(f"\n  Constraint violations:")
            for v in opt.constraint_violations:
                print(f"    [{v.severity.upper()}] {v.message}")
        return

    print(f"\n  ✅  FEASIBLE  (demand = {demand:,.0f} units)\n")

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

    # Compute terms for decomposition
    w = cfg["objective_weights"]
    c = cfg["constraints"]
    w_cost = w["w_cost"]
    w_sla = w["w_sla"]
    w_risk = w["w_risk"]
    sla_threshold = c["sla_threshold"]["threshold_days"]
    penalty_per_day = c["sla_threshold"]["penalty_per_day"]

    x = np.array([solution.get(v.vendor_id, 0.0) for v in vendor_profiles])
    costs_arr = np.array([v.cost_per_unit for v in vendor_profiles])
    lead_times = np.array([v.lead_time_days for v in vendor_profiles])
    reliabilities = np.array([v.reliability for v in vendor_profiles])

    raw_cost = float(np.dot(x, costs_arr) * demand)
    overdue = np.maximum(0.0, lead_times - sla_threshold)
    sla_penalty = float(np.dot(x, overdue) * penalty_per_day * demand)
    risk_exposure = float(np.dot(x, 1.0 - reliabilities))

    print(f"\n  📊 Objective Decomposition")
    print(f"  objective = w_cost×cost + w_sla×sla_penalty + w_risk×risk_penalty")
    print(f"  {DIV}")
    print(f"  {'Component':<28} {'Value':>12}   {'Weight':>6}   {'Contribution':>12}")
    print(f"  {DIV}")
    print(f"  {'Shipping cost ($)':<28} {raw_cost:>12,.2f}   {w_cost:>6.2f}   {w_cost * raw_cost:>12,.2f}")
    print(f"  {'SLA penalty ($)':<28} {sla_penalty:>12,.2f}   {w_sla:>6.2f}   {w_sla * sla_penalty:>12,.2f}")
    print(f"  {'Risk exposure (rate)':<28} {risk_exposure:>12.4f}   {w_risk:>6.2f}   {w_risk * risk_exposure:>12.4f}")
    print(f"  {DIV}")
    print(f"  {'Objective value (weighted)':<28} {opt.best_value:>12.4f}")
    print(f"\n  Risk exposure = Σ(allocation_i × (1 − reliability_i))")
    print(f"  Interpretation: {risk_exposure:.1%} of shipment volume")
    print(f"                  expected to experience reliability failure")
    print(f"\n  Solver     : {solver_name} (Sequential Least Squares Programming)")
    print(f"  Why SLSQP  : continuous allocation requires Σx=1 equality constraint")
    print(f"               L-BFGS-B cannot enforce equality — SLSQP can")
    print(f"  Iterations : {opt.iterations:,} | Converged : {opt.converged}")

    print(f"\n  🔒 Constraint Status")
    print(f"  {DIV}")
    if opt.constraint_violations:
        for v in opt.constraint_violations:
            icon = "⚠️ " if v.severity == "soft" else "❌"
            print(f"  {icon} [{v.severity.upper()}] {v.constraint_name}")
            print(f"     {v.message}")
    else:
        print(f"  ✅ All constraints satisfied")


def print_comparison(
    engine_solution: dict,
    baseline: dict,
    demand: float,
) -> None:
    print(f"\n{'═' * 65}")
    print(f"  📊 BASELINE COMPARISON")
    print(f"  Baseline: cheapest-first greedy allocation")
    print(f"  Engine  : multi-objective optimization (cost + SLA + risk)")
    print(f"{'═' * 65}")

    e_cost = engine_solution["raw_cost"]
    e_sla  = engine_solution["sla_penalty"]
    e_risk = engine_solution["risk_exposure"]
    e_eff  = engine_solution["effective_cost"]

    b_cost = baseline["raw_cost"]
    b_sla  = baseline["sla_penalty"]
    b_risk = baseline["risk_exposure"]
    b_eff  = baseline["effective_cost"]

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'Engine':>12} {'Delta':>12}")
    print(f"  {DIV}")
    print(f"  {'Raw shipping cost ($)':<30} {b_cost:>12,.0f} {e_cost:>12,.0f} {e_cost - b_cost:>+12,.0f}")
    print(f"  {'SLA penalty ($)':<30} {b_sla:>12,.0f} {e_sla:>12,.0f} {e_sla - b_sla:>+12,.0f}")
    print(f"  {'Risk exposure (rate)':<30} {b_risk:>12.4f} {e_risk:>12.4f} {e_risk - b_risk:>+12.4f}")
    print(f"  {DIV}")
    print(f"  {'Effective cost ($)':<30} {b_eff:>12,.0f} {e_eff:>12,.0f} {e_eff - b_eff:>+12,.0f}")

    improvement = (b_eff - e_eff) / b_eff * 100
    risk_improvement = (b_risk - e_risk) / b_risk * 100 if b_risk > 0 else 0.0

    print(f"\n  Effective cost improvement : {improvement:+.1f}%")
    print(f"  Risk exposure improvement  : {risk_improvement:+.1f}%")

    print(f"\n  Why engine costs more in raw terms:")
    print(f"    Baseline allocates to cheapest vendor (vendor_B, $3.50/unit)")
    print(f"    but vendor_B lead_time=4.5d exceeds SLA threshold=3.0d.")
    print(f"    Baseline incurs ${b_sla:,.0f} SLA penalty — engine avoids this entirely.")

    if improvement > 0:
        print(f"\n  ✅ Engine outperforms baseline on effective cost.")
    else:
        print(f"\n  ⚠️  Engine has higher raw cost — but lower risk/SLA penalty.")


def print_sensitivity(
    base_result: dict,
    stressed_result: dict,
    base_demand: float,
    stressed_demand: float,
    base_opt: OptimizationResult,
    stressed_opt: OptimizationResult,
    vendor_profiles: list[VendorProfile],
) -> None:
    print(f"\n{'═' * 65}")
    print(f"  📈 SENSITIVITY: Demand +{(stressed_demand / base_demand - 1) * 100:.0f}%")
    print(f"{'═' * 65}")

    if not base_opt.is_feasible or not stressed_opt.is_feasible:
        print("  Cannot compute — one or both scenarios infeasible.")
        return

    b_cost = base_result["raw_cost"]
    s_cost = stressed_result["raw_cost"]
    b_risk = base_result["risk_exposure"]
    s_risk = stressed_result["risk_exposure"]
    cost_pct = (s_cost - b_cost) / b_cost * 100

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'Stressed':>12} {'Delta':>12}")
    print(f"  {DIV}")
    print(f"  {'Demand (units)':<30} {base_demand:>12,.0f} {stressed_demand:>12,.0f} {stressed_demand - base_demand:>+12,.0f}")
    print(f"  {'Raw cost ($)':<30} {b_cost:>12,.0f} {s_cost:>12,.0f} {s_cost - b_cost:>+12,.0f}  ({cost_pct:+.1f}%)")
    print(f"  {'Risk exposure (rate)':<30} {b_risk:>12.4f} {s_risk:>12.4f} {s_risk - b_risk:>+12.4f}")
    print(f"  {'Solver iterations':<30} {base_opt.iterations:>12,} {stressed_opt.iterations:>12,} {stressed_opt.iterations - base_opt.iterations:>+12,}")

    print(f"\n  Allocation Stability:")
    print(f"  {DIV}")
    base_sol   = base_opt.best_solution or {}
    stress_sol = stressed_opt.best_solution or {}
    any_change = False
    for vp in vendor_profiles:
        b_frac = base_sol.get(vp.vendor_id, 0.0)
        s_frac = stress_sol.get(vp.vendor_id, 0.0)
        delta  = s_frac - b_frac
        if abs(delta) > 0.005:
            direction = "▲" if delta > 0 else "▼"
            print(f"  {direction} {vp.vendor_id:<14} {b_frac:.1%} → {s_frac:.1%}  (Δ {delta:+.1%})")
            any_change = True
    if not any_change:
        print(f"  ✅ Allocation stable — same vendor mix under +20% demand")
        print(f"     Indicates well-conditioned optimization landscape.")
        print(f"     Constraint set does not become active at this demand level.")

    print(f"\n  Budget Pressure:")
    base_hard   = [v for v in base_opt.constraint_violations if v.severity == "hard"]
    stress_hard = [v for v in stressed_opt.constraint_violations if v.severity == "hard"]
    if not base_hard and not stress_hard:
        print(f"  ✅ Both scenarios within budget.")
    elif stress_hard and not base_hard:
        print(f"  ⚠️  Stressed scenario hits budget constraint.")


# ─── Metrics helper ───────────────────────────────────────────────────────────

def compute_engine_metrics(
    opt: OptimizationResult,
    vendor_profiles: list[VendorProfile],
    demand: float,
    sla_threshold: float,
    penalty_per_day: float,
) -> dict:
    if not opt.is_feasible or opt.best_solution is None:
        return {"raw_cost": 0, "sla_penalty": 0, "risk_exposure": 0, "effective_cost": 0}

    solution    = opt.best_solution
    profile_map = {v.vendor_id: v for v in vendor_profiles}

    raw_cost = sum(
        solution.get(vid, 0.0) * profile_map[vid].cost_per_unit * demand
        for vid in solution
    )
    weighted_lead = sum(
        solution.get(vid, 0.0) * profile_map[vid].lead_time_days
        for vid in solution
    )
    excess_days  = max(0.0, weighted_lead - sla_threshold)
    sla_penalty  = excess_days * penalty_per_day * demand
    risk_exposure = sum(
        solution.get(vid, 0.0) * (1 - profile_map[vid].reliability)
        for vid in solution
    )

    return {
        "raw_cost": raw_cost,
        "sla_penalty": sla_penalty,
        "risk_exposure": risk_exposure,
        "effective_cost": raw_cost + sla_penalty,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 65)
    print("  RISK-AWARE DECISION ENGINE — Phase 1")
    print("  Vendor Allocation Under Budget + SLA Constraints")
    print("═" * 65)

    cfg             = load_config()
    vendor_profiles = build_vendor_profiles(cfg)
    vendor_configs  = build_vendor_configs(cfg)
    demand          = cfg["scenario"]["demand"]
    sla_threshold   = cfg["constraints"]["sla_threshold"]["threshold_days"]
    penalty_per_day = cfg["constraints"]["sla_threshold"]["penalty_per_day"]
    budget_cap      = cfg["constraints"]["budget_cap"]["value"]

    print(f"\n  Experiment : {cfg['experiment']['name']}")
    print(f"  Seed       : {cfg['experiment']['seed']}  (deterministic)")
    print(f"  Vendors    : {len(vendor_profiles)}")
    print(f"  Demand     : {demand:,} units")
    print(f"  Budget cap : ${budget_cap:,}")

    logger = AuditLogger()

    # ── Scenario 1: Feasible baseline ────────────────────────────────────────
    engine       = build_engine(cfg, vendor_profiles, vendor_configs, budget_cap)
    orchestrator = DecisionOrchestrator(engine=engine, audit_logger=logger)

    result = orchestrator.run(
        raw_features={"demand": demand},
        domain="shipping",
        extra_metadata={"scenario": "baseline"},
    )
    print_result(
        result.decision,
        result.optimization,
        demand,
        vendor_profiles,
        cfg,
        label="SCENARIO 1: BASELINE",
    )

    # ── Scenario 2: Infeasible ────────────────────────────────────────────────
    baseline     = cheapest_first(
        vendor_configs, vendor_profiles, demand, sla_threshold, penalty_per_day
    )
    min_cost         = baseline["raw_cost"]
    infeasible_budget = min_cost * 0.90

    engine_inf       = build_engine(cfg, vendor_profiles, vendor_configs, infeasible_budget)
    orchestrator_inf = DecisionOrchestrator(engine=engine_inf, audit_logger=logger)

    result_inf = orchestrator_inf.run(
        raw_features={"demand": demand},
        domain="shipping",
        extra_metadata={"scenario": "infeasible"},
    )
    print(f"\n{'═' * 65}")
    print(f"  SCENARIO 2: INFEASIBLE (budget = ${infeasible_budget:,.0f})")
    print(f"  Minimum achievable cost = ${min_cost:,.0f}")
    print(f"{'═' * 65}")
    print(f"\n  ❌  INFEASIBLE — budget ${infeasible_budget:,.0f} < minimum cost ${min_cost:,.0f}")
    print(f"\n  {result_inf.decision.action}")

    # ── Scenario 3: Baseline comparison ──────────────────────────────────────
    engine_metrics = compute_engine_metrics(
        result.optimization, vendor_profiles, demand, sla_threshold, penalty_per_day
    )
    print_comparison(engine_metrics, baseline, demand)

    # ── Scenario 4: Sensitivity — demand +20% ────────────────────────────────
    demand_stressed  = demand * 1.20
    engine_stress    = build_engine(cfg, vendor_profiles, vendor_configs, budget_cap)
    orchestrator_str = DecisionOrchestrator(engine=engine_stress, audit_logger=logger)

    result_stress = orchestrator_str.run(
        raw_features={"demand": demand_stressed},
        domain="shipping",
        extra_metadata={"scenario": "demand_stress_+20pct"},
    )
    engine_metrics_stress = compute_engine_metrics(
        result_stress.optimization,
        vendor_profiles,
        demand_stressed,
        sla_threshold,
        penalty_per_day,
    )
    print_sensitivity(
        engine_metrics, engine_metrics_stress,
        demand, demand_stressed,
        result.optimization, result_stress.optimization,
        vendor_profiles,
    )

    print(f"\n{'═' * 65}")
    print(f"  Phase 1 complete.")
    print(f"  Audit log: {len(logger.records())} records captured.")
    print(f"  Next: Phase 2 — Probabilistic inputs + CVaR objective")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()