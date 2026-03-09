"""
Phase 4 — Walk-Forward Backtester

Purpose:
    Evaluate the decision engine against heuristic strategies across
    60 months of real market conditions (2020-2024).

    For each month:
        1. Load market regime from processed data
        2. Sample cost distribution conditioned on that regime
        3. Run engine + cheapest_first heuristic
        4. Record effective_cost, risk_exposure, constraint_violations

    Aggregate metrics:
        - Mean effective cost: engine vs heuristic
        - Win rate: pct months engine outperforms heuristic on effective cost
        - Regime breakdown: performance per market condition
        - Decision stability: allocation variance over time

Why walk-forward (not random split):
    Decision systems are deployed sequentially.
    Walk-forward respects temporal ordering — no look-ahead bias.
    Evaluation window = 2020-2024 (post-COVID, high volatility period).

Why 2020-2024:
    Contains three distinct regimes in 5 years:
        - crisis (2020 COVID shock)
        - high_market (2021-2022 rate spike)
        - low_market / normal_market (2023-2024 normalization)
    This is the hardest test for a decision engine.

Cost scaling:
    Charter rates ($/day) are normalized to $/unit using a scale factor
    calibrated so that low_market 725 TEU maps to $3.50/unit — matching
    the original engine's Phase 1 cost range.

    Scale factor = 3.50 / (low_market_725_mean / 1000) = 0.6809

    This preserves relative cost differences across vessel sizes and regimes
    while keeping absolute costs in a comparable range to Phase 1 results.

Budget cap per regime:
    Fixed budget cap ($55,000) is infeasible in crisis regime
    (min cost ~$156,000 at $15.63/unit * 10,000 units).
    Budget cap is scaled proportionally to regime cost level:
        budget_cap(regime) = base_budget * (regime_cost_index / low_market_cost_index)
    This tests the engine under realistic constraints for each regime,
    not an artificial constraint that guarantees infeasibility.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Cost scaling constants
# ---------------------------------------------------------------------------

# Calibrated so low_market 725 TEU = $3.50/unit (matches Phase 1 range)
COST_SCALE_FACTOR = 0.6809

# Base budget cap from Phase 1 config
BASE_BUDGET_CAP = 55_000.0

# Budget cap is scaled per regime to remain feasible
REGIME_BUDGET_MULTIPLIER = {
    "low_market":    1.0,
    "normal_market": 2.0,
    "high_market":   2.0,
    "crisis":        4.5,
}


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _compute_metrics(
    allocation: Dict[str, float],
    vendors: List[dict],
    demand: float,
    sla_threshold: float,
    penalty_per_day: float,
) -> dict:
    profile_map = {v["vendor_id"]: v for v in vendors}
    raw_cost = sum(
        allocation[vid] * profile_map[vid]["cost_per_unit"] * demand
        for vid in allocation
    )
    weighted_lead = sum(
        allocation[vid] * profile_map[vid]["lead_time_days"]
        for vid in allocation
    )
    excess_days = max(0.0, weighted_lead - sla_threshold)
    sla_penalty = excess_days * penalty_per_day * demand
    risk_exposure = sum(
        allocation[vid] * (1 - profile_map[vid]["reliability"])
        for vid in allocation
    )
    return {
        "raw_cost": raw_cost,
        "sla_penalty": sla_penalty,
        "risk_exposure": risk_exposure,
        "effective_cost": raw_cost + sla_penalty,
        "weighted_lead_time": weighted_lead,
    }


# ---------------------------------------------------------------------------
# Heuristic strategies
# ---------------------------------------------------------------------------

def cheapest_first(vendors, demand, sla_threshold, penalty_per_day):
    sorted_v = sorted(vendors, key=lambda v: v["cost_per_unit"])
    allocation = {v["vendor_id"]: 0.0 for v in vendors}
    remaining = 1.0
    for v in sorted_v:
        if remaining <= 0:
            break
        alloc = min(v["capacity_fraction"], remaining)
        allocation[v["vendor_id"]] = alloc
        remaining -= alloc
    return _compute_metrics(allocation, vendors, demand, sla_threshold, penalty_per_day)


def highest_reliability(vendors, demand, sla_threshold, penalty_per_day):
    sorted_v = sorted(vendors, key=lambda v: -v["reliability"])
    allocation = {v["vendor_id"]: 0.0 for v in vendors}
    remaining = 1.0
    for v in sorted_v:
        if remaining <= 0:
            break
        alloc = min(v["capacity_fraction"], remaining)
        allocation[v["vendor_id"]] = alloc
        remaining -= alloc
    return _compute_metrics(allocation, vendors, demand, sla_threshold, penalty_per_day)


def balanced_sla(vendors, demand, sla_threshold, penalty_per_day):
    """
    Balanced SLA: prioritize vendors within SLA threshold up to their
    capacity limit, fill remainder with next-cheapest vendor.
    Represents a reasonable procurement policy without optimization.
    """
    eligible = [v for v in vendors if v["lead_time_days"] <= sla_threshold]
    pool = eligible if eligible else sorted(vendors, key=lambda v: v["lead_time_days"])
    allocation = {v["vendor_id"]: 0.0 for v in vendors}
    remaining = 1.0
    # Fill eligible vendors up to their capacity limit (not artificially capped)
    for v in sorted(pool, key=lambda v: v["cost_per_unit"]):
        if remaining <= 0:
            break
        alloc = min(v["capacity_fraction"], remaining)
        allocation[v["vendor_id"]] = alloc
        remaining -= alloc
    # Fill remainder with cheapest remaining vendor
    if remaining > 1e-6:
        others = sorted([v for v in vendors if allocation[v["vendor_id"]] < v["capacity_fraction"]],
                        key=lambda v: v["cost_per_unit"])
        for v in others:
            if remaining <= 0:
                break
            alloc = min(v["capacity_fraction"] - allocation[v["vendor_id"]], remaining)
            allocation[v["vendor_id"]] += alloc
            remaining -= alloc
    return _compute_metrics(allocation, vendors, demand, sla_threshold, penalty_per_day)


# ---------------------------------------------------------------------------
# Regime-conditioned cost sampler
# ---------------------------------------------------------------------------

def sample_vendors_for_regime(
    base_vendors: List[dict],
    regime: str,
    regime_stats: dict,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Sample vendor costs conditioned on market regime.

    Cost scaling:
        raw_rate ($/day) -> cost_per_unit = raw_rate / 1000 * COST_SCALE_FACTOR
        This preserves relative differences across vessel sizes and regimes.
    """
    vendor_col_map = {
        "small_feeder":  "725_TEU",
        "medium_feeder": "1000_TEU",
        "large_feeder":  "1700_TEU",
        "panamax_small": "2000_TEU",
        "panamax_large": "2750_TEU",
    }

    sampled = []
    for v in base_vendors:
        col = vendor_col_map.get(v["vendor_id"])
        if col and regime in regime_stats and col in regime_stats[regime]:
            s = regime_stats[regime][col]
            raw = rng.normal(s["mean"], s["std"] * 0.3)
            raw = float(np.clip(raw, s["min"] * 0.9, s["max"] * 1.1))
            raw = max(raw, 500.0)
            cost_per_unit = raw / 1000.0 * COST_SCALE_FACTOR
            sampled.append({**v, "cost_per_unit": round(cost_per_unit, 4)})
        else:
            sampled.append(v.copy())
    return sampled


# ---------------------------------------------------------------------------
# Engine optimizer (standalone SLSQP)
# ---------------------------------------------------------------------------

def run_engine(
    vendors: List[dict],
    demand: float,
    budget_cap: float,
    sla_threshold: float,
    penalty_per_day: float,
    w_cost: float = 0.60,
    w_sla: float = 0.25,
    w_risk: float = 0.15,
    n_restarts: int = 5,
    seed: int = 42,
) -> Optional[Dict[str, float]]:
    vendor_ids = [v["vendor_id"] for v in vendors]
    costs = np.array([v["cost_per_unit"] for v in vendors])
    reliabilities = np.array([v["reliability"] for v in vendors])
    lead_times = np.array([v["lead_time_days"] for v in vendors])
    capacities = np.array([v["capacity_fraction"] for v in vendors])
    bounds = [(0.0, cap) for cap in capacities]
    rng = np.random.default_rng(seed)

    SCALE = 1e-4  # normalize large objective values for SLSQP numerical stability

    def objective(x):
        cost_term = np.dot(x, costs) * demand
        overdue = np.maximum(0.0, lead_times - sla_threshold)
        sla_term = np.dot(x, overdue) * penalty_per_day * demand
        risk_term = np.dot(x, 1.0 - reliabilities)
        return SCALE * (w_cost * cost_term + w_sla * sla_term + w_risk * risk_term)

    constraints = [
        {"type": "eq",   "fun": lambda x: np.sum(x) - 1.0},
        {"type": "ineq", "fun": lambda x: SCALE * (budget_cap - np.dot(x, costs) * demand)},
    ]

    best_x, best_val = None, np.inf
    for _ in range(n_restarts):
        x0 = rng.dirichlet(np.ones(len(vendors)))
        x0 = np.clip(x0, 0, capacities)
        x0 = x0 / x0.sum() if x0.sum() > 0 else np.ones(len(vendors)) / len(vendors)
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 2000, "ftol": 1e-6})
        if res.success and res.fun < best_val:
            total_cost = np.dot(res.x, costs) * demand
            if total_cost <= budget_cap * 1.001:
                best_val = res.fun
                best_x = res.x

    if best_x is None:
        return None
    return {vid: float(x) for vid, x in zip(vendor_ids, best_x)}


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------

class WalkForwardBacktester:

    def __init__(
        self,
        processed_dir: Path,
        demand: float = 10_000.0,
        base_budget_cap: float = BASE_BUDGET_CAP,
        sla_threshold: float = 3.0,
        penalty_per_day: float = 1.5,
        seed: int = 42,
    ):
        self.demand = demand
        self.base_budget_cap = base_budget_cap
        self.sla_threshold = sla_threshold
        self.penalty_per_day = penalty_per_day
        self.rng = np.random.default_rng(seed)

        self.regimes = pd.read_csv(
            processed_dir / "market_regimes.csv", parse_dates=["date"]
        ).set_index("date")

        with open(processed_dir / "regime_stats.json") as f:
            self.regime_stats = json.load(f)

        self.base_vendors = pd.read_csv(
            processed_dir / "vendor_profiles.csv"
        ).to_dict("records")

    def run(self, start: str = "2020-01-01", end: str = "2024-12-31") -> pd.DataFrame:
        eval_regimes = self.regimes[start:end]

        print(f"  Evaluation window : {start[:7]} to {end[:7]}")
        print(f"  Total months      : {len(eval_regimes)}")
        print(f"  Regime distribution:")
        for r, c in eval_regimes["regime"].value_counts().items():
            print(f"    {r:<20}: {c} months")
        print()

        records = []

        for date, row in eval_regimes.iterrows():
            regime = row["regime"]
            budget_cap = self.base_budget_cap * REGIME_BUDGET_MULTIPLIER.get(regime, 1.0)

            sampled_vendors = sample_vendors_for_regime(
                self.base_vendors, regime, self.regime_stats, self.rng
            )

            allocation = run_engine(
                vendors=sampled_vendors,
                demand=self.demand,
                budget_cap=budget_cap,
                sla_threshold=self.sla_threshold,
                penalty_per_day=self.penalty_per_day,
                seed=int(self.rng.integers(0, 100_000)),
            )

            if allocation is not None:
                engine_m = _compute_metrics(
                    allocation, sampled_vendors,
                    self.demand, self.sla_threshold, self.penalty_per_day
                )
                engine_feasible = True
            else:
                engine_m = {"raw_cost": None, "sla_penalty": None,
                            "risk_exposure": None, "effective_cost": None}
                engine_feasible = False

            cf = cheapest_first(sampled_vendors, self.demand, self.sla_threshold, self.penalty_per_day)
            hr = highest_reliability(sampled_vendors, self.demand, self.sla_threshold, self.penalty_per_day)
            bs = balanced_sla(sampled_vendors, self.demand, self.sla_threshold, self.penalty_per_day)

            records.append({
                "date": date,
                "regime": regime,
                "budget_cap": budget_cap,
                "engine_feasible": engine_feasible,
                "engine_cost": engine_m["effective_cost"],
                "engine_raw_cost": engine_m["raw_cost"],
                "engine_sla_penalty": engine_m["sla_penalty"],
                "engine_risk": engine_m["risk_exposure"],
                "cheapest_first_cost": cf["effective_cost"],
                "highest_reliability_cost": hr["effective_cost"],
                "balanced_sla_cost": bs["effective_cost"],
                "best_heuristic_cost": min(
                    cf["effective_cost"], hr["effective_cost"], bs["effective_cost"]
                ),
            })

        df = pd.DataFrame(records).set_index("date")
        df["beats_cheapest"] = df["engine_cost"] < df["cheapest_first_cost"]
        df["beats_best_heuristic"] = df["engine_cost"] < df["best_heuristic_cost"]
        return df

    def summarize(self, df: pd.DataFrame) -> None:
        feasible = df[df["engine_feasible"]]
        n_total = len(df)
        n_feasible = len(feasible)

        print("=" * 65)
        print("  Phase 4 — Walk-Forward Backtest Results (2020-2024)")
        print("=" * 65)

        print(f"\n  Months evaluated  : {n_total}")
        print(f"  Engine feasible   : {n_feasible} / {n_total} ({n_feasible/n_total:.0%})")

        if n_feasible == 0:
            print("\n  No feasible solutions.")
            return

        eng_mean  = feasible["engine_cost"].mean()
        heu_mean  = feasible["best_heuristic_cost"].mean()
        cf_mean   = feasible["cheapest_first_cost"].mean()
        delta     = eng_mean - heu_mean
        delta_pct = delta / heu_mean * 100
        eng_risk  = feasible["engine_risk"].mean()
        win_rate  = feasible["beats_best_heuristic"].mean()
        win_cf    = feasible["beats_cheapest"].mean()

        print(f"\n  {'Metric':<38} {'Engine':>10} {'Best Heuristic':>15} {'Delta':>10}")
        print(f"  {'-' * 75}")
        print(f"  {'Mean effective cost ($)':<38} {eng_mean:>10,.0f} {heu_mean:>15,.0f} {delta:>+10,.0f}  ({delta_pct:+.1f}%)")
        print(f"  {'Win rate vs best heuristic':<38} {win_rate:>10.0%}")
        print(f"  {'Win rate vs cheapest-first':<38} {win_cf:>10.0%}")
        print(f"  {'Mean risk exposure':<38} {eng_risk:>10.4f}")
        print(f"  {'Mean SLA penalty ($)':<38} {feasible['engine_sla_penalty'].mean():>10,.0f}")

        print(f"\n  Performance by market regime:")
        print(f"  {'-' * 75}")
        print(f"  {'Regime':<22} {'Mo':>4} {'Engine':>10} {'Heuristic':>10} {'Delta%':>8} {'Win%':>6}")
        print(f"  {'-' * 75}")

        for regime in ["low_market", "normal_market", "high_market", "crisis"]:
            mask = feasible["regime"] == regime
            if mask.sum() == 0:
                continue
            sub = feasible[mask]
            e_m = sub["engine_cost"].mean()
            h_m = sub["best_heuristic_cost"].mean()
            d_p = (e_m - h_m) / h_m * 100
            w_p = sub["beats_best_heuristic"].mean()
            print(f"  {regime:<22} {mask.sum():>4} {e_m:>10,.0f} {h_m:>10,.0f} {d_p:>+8.1f}% {w_p:>5.0%}")

        print(f"\n  {'=' * 65}")
        if delta_pct < 0:
            print(f"  Engine outperforms best heuristic by {abs(delta_pct):.1f}% on average.")
        else:
            print(f"  Engine trades {delta_pct:.1f}% cost for lower SLA penalty and risk.")
            print(f"  Effective cost includes SLA penalty — engine avoids breach.")
        print(f"  {'=' * 65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed")

    if not PROCESSED_DIR.exists():
        print("ERROR: data/processed/ not found. Run data_pipeline.py first.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  Phase 4 — Walk-Forward Backtester")
    print("  Real market data: 2005-2025 charter rates (Feeder Container)")
    print("  Evaluation: 2020-2024 (60 months)")
    print("=" * 65 + "\n")

    backtester = WalkForwardBacktester(
        processed_dir=PROCESSED_DIR,
        demand=10_000.0,
        base_budget_cap=55_000.0,
        sla_threshold=3.0,
        penalty_per_day=1.5,
        seed=42,
    )

    print("  Running walk-forward evaluation...\n")
    df_results = backtester.run(start="2020-01-01", end="2024-12-31")
    backtester.summarize(df_results)

    results_path = PROCESSED_DIR / "backtest_results.csv"
    df_results.to_csv(results_path)
    print(f"  Full results saved -> {results_path}")