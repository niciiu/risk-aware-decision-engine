"""
MonteCarloRiskEvaluator — domain-agnostic risk quantification.

Purpose:
    Given a fixed allocation (decision already made by optimizer),
    evaluate how risky that decision is under demand uncertainty.

Why separate from optimizer:
    Optimizer answers: "what is the best allocation?"
    Risk evaluator answers: "how bad can this allocation get?"
    These are different questions with different math.

    Mixing them produces an optimizer that is risk-aware but slow.
    Separating them produces a fast optimizer + an auditable risk layer.

Mathematical definitions:
    Expected cost  = E[cost(x, D)]        where D ~ demand_distribution
    VaR-95         = 95th percentile of cost distribution
    CVaR-95        = E[cost | cost > VaR-95]  (expected cost in worst 5%)
    P(budget)      = P(cost(x, D) > budget_cap)
    P(SLA breach)  = P(weighted_lead_time > sla_threshold)

Why CVaR over VaR:
    VaR tells you the threshold. CVaR tells you what happens beyond it.
    For operational decisions, the magnitude of bad outcomes matters
    more than just knowing they exist.
    CVaR is also convex — makes it usable as an objective in Phase 2+.

Business ownership:
    Risk evaluation is owned by Risk team / CFO office.
    They set tolerance thresholds — not Engineering.

Phase 2 note:
    This class is called AFTER the optimizer in Phase 2.
    Phase 3 extension: use CVaR as the objective itself (not post-hoc).
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np

from decision_intelligence.core.risk.base import BaseRiskModel, RiskReport


@dataclass
class DemandDistribution:
    """
    Truncated Normal demand distribution.

    Why truncated (not plain Normal):
        Demand cannot be negative. Plain Normal can sample negative values
        at the tail, which is physically meaningless and breaks cost calculations.

    Parameters:
        mu    : expected demand (units)
        sigma : standard deviation (units)
        lower : minimum demand (default 0 — no negative demand)
        upper : maximum demand (default inf — no hard cap)

    CV = sigma / mu = 15% for our baseline.
    This reflects typical weekly/monthly demand uncertainty
    in supply chain literature.
    """
    mu: float
    sigma: float
    lower: float = 0.0
    upper: float = float("inf")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample n demand values from truncated Normal.

        Method: rejection sampling.
        Draw from Normal, reject outside [lower, upper], repeat until n valid samples.
        For CV=15% and lower=0, rejection rate is negligible (<0.001%).
        """
        samples = np.empty(n)
        filled = 0
        while filled < n:
            needed = n - filled
            # Oversample to reduce iterations
            candidates = rng.normal(self.mu, self.sigma, size=needed * 2)
            valid = candidates[(candidates >= self.lower) & (candidates <= self.upper)]
            take = min(len(valid), needed)
            samples[filled:filled + take] = valid[:take]
            filled += take
        return samples


class MonteCarloRiskEvaluator(BaseRiskModel):
    """
    Domain-agnostic Monte Carlo risk evaluator.

    Subclasses implement `compute_cost()` with domain-specific logic.
    This class handles all simulation mechanics.

    Config keys:
        n_simulations : int   — number of Monte Carlo samples (default 10000)
        seed          : int   — RNG seed for reproducibility (default 42)
        budget_cap    : float — hard budget threshold for P(exceed) calculation
        cvar_alpha    : float — tail probability for CVaR (default 0.05 = CVaR-95)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        demand_distribution: DemandDistribution,
    ) -> None:
        super().__init__(config)
        self.distribution = demand_distribution
        self.n_simulations: int = config.get("n_simulations", 10_000)
        self.seed: int = config.get("seed", 42)
        self.budget_cap: float = config.get("budget_cap", float("inf"))
        self.cvar_alpha: float = config.get("cvar_alpha", 0.05)  # CVaR-95 = top 5%

    @abstractmethod
    def compute_cost(
        self,
        solution: Any,
        demand: float,
        context: Dict[str, Any],
    ) -> float:
        """
        Compute total cost for a given allocation and demand sample.

        Subclasses implement domain-specific cost calculation here.
        Called N times — must be fast (no solver calls).

        Args:
            solution : allocation dict from optimizer
            demand   : sampled demand value for this simulation
            context  : original decision context

        Returns:
            scalar cost value for this scenario
        """
        ...

    @abstractmethod
    def compute_sla_breach(
        self,
        solution: Any,
        demand: float,
        context: Dict[str, Any],
    ) -> bool:
        """
        Return True if this allocation breaches SLA for given demand sample.

        Subclasses implement domain-specific SLA check here.
        """
        ...

    def assess(
        self,
        solution: Any,
        context: Dict[str, Any],
    ) -> RiskReport:
        """
        Run Monte Carlo simulation and return full risk report.

        Pipeline:
            1. Sample N demand scenarios from distribution
            2. Compute cost for each scenario
            3. Compute SLA breach indicator for each scenario
            4. Aggregate into risk metrics

        Deterministic: same seed → same result always.
        """
        rng = np.random.default_rng(self.seed)

        # Step 1: Sample demand scenarios
        demand_samples = self.distribution.sample(self.n_simulations, rng)

        # Step 2: Compute cost for each scenario
        costs = np.array([
            self.compute_cost(solution, float(d), context)
            for d in demand_samples
        ])

        # Step 3: Compute SLA breach for each scenario
        sla_breaches = np.array([
            self.compute_sla_breach(solution, float(d), context)
            for d in demand_samples
        ])

        # Step 4: Aggregate metrics
        expected_cost = float(np.mean(costs))
        cost_std = float(np.std(costs))

        # VaR-95: 95th percentile of cost distribution
        var_95 = float(np.percentile(costs, 95))

        # CVaR-95: mean of worst 5% scenarios
        tail_mask = costs >= var_95
        cvar_95 = float(np.mean(costs[tail_mask])) if tail_mask.any() else var_95

        # P(budget exceed): fraction of scenarios where cost > budget_cap
        p_budget_exceed = float(np.mean(costs > self.budget_cap))

        # P(SLA breach): fraction of scenarios with SLA violation
        p_sla_breach = float(np.mean(sla_breaches))

        return RiskReport(
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=float(np.max(costs) - np.min(costs)),
            volatility=cost_std,
            is_within_tolerance=(
                cvar_95 <= self.budget_cap
                and p_budget_exceed <= self.config.get("max_p_budget_exceed", 0.10)
            ),
            extra_metrics={
                "expected_cost": expected_cost,
                "cost_std": cost_std,
                "p_budget_exceed": p_budget_exceed,
                "p_sla_breach": p_sla_breach,
                "n_simulations": float(self.n_simulations),
                "demand_mu": self.distribution.mu,
                "demand_sigma": self.distribution.sigma,
            },
        )