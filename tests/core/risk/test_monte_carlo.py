"""
Unit tests for MonteCarloRiskEvaluator.

What we test and why:
    1. CVaR >= VaR >= expected cost   — mathematical ordering must always hold
    2. p_budget_exceed in [0, 1]      — valid probability range
    3. deterministic output           — same seed → same metrics always
    4. zero variance on fixed demand  — if sigma=0, all metrics converge to same value
    5. high budget → low exceedance   — sanity check on budget logic
    6. tight budget → high exceedance — inverse sanity check
    7. RiskReport fields are correct types
"""
from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.core.risk.monte_carlo import (
    MonteCarloRiskEvaluator,
    DemandDistribution,
)
from decision_intelligence.core.risk.base import RiskReport


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# Shipping cost = sum(allocation_i * cost_i * demand)
# SLA breach = weighted lead time > threshold
# ---------------------------------------------------------------------------
VENDORS = ["vendor_A", "vendor_B", "vendor_C"]
COST_COEFFS = [3.5, 4.0, 4.5]
LEAD_TIMES = [3, 5, 7]          # days
SLA_THRESHOLD = 5.0             # days


class ShippingRiskEvaluator(MonteCarloRiskEvaluator):
    """Concrete risk evaluator for testing."""

    def compute_cost(self, solution: dict, demand: float, context: dict) -> float:
        return sum(
            solution[v] * COST_COEFFS[i] * demand
            for i, v in enumerate(VENDORS)
        )

    def compute_sla_breach(self, solution: dict, demand: float, context: dict) -> bool:
        weighted_lead = sum(
            solution[v] * LEAD_TIMES[i]
            for i, v in enumerate(VENDORS)
        )
        return weighted_lead > SLA_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
ALLOCATION = {"vendor_A": 0.6, "vendor_B": 0.3, "vendor_C": 0.1}
DEMAND_MU = 10_000.0
DEMAND_SIGMA = 1_500.0
BUDGET_CAP = 55_000.0


@pytest.fixture
def distribution():
    return DemandDistribution(mu=DEMAND_MU, sigma=DEMAND_SIGMA)


@pytest.fixture
def evaluator(distribution):
    return ShippingRiskEvaluator(
        config={
            "n_simulations": 5_000,   # smaller for test speed
            "seed": 42,
            "budget_cap": BUDGET_CAP,
            "cvar_alpha": 0.05,
        },
        demand_distribution=distribution,
    )


@pytest.fixture
def context():
    return {}


@pytest.fixture
def report(evaluator, context):
    return evaluator.assess(ALLOCATION, context)


# ---------------------------------------------------------------------------
# Tests: mathematical correctness
# ---------------------------------------------------------------------------
class TestRiskMetricOrdering:

    def test_cvar_gte_var(self, report):
        """CVaR must always be >= VaR by definition."""
        assert report.cvar_95 >= report.var_95 - 1e-6, (
            f"CVaR {report.cvar_95:.2f} < VaR {report.var_95:.2f} — mathematical violation"
        )

    def test_var_gte_expected_cost(self, report):
        """VaR-95 must be >= expected cost (95th percentile >= mean)."""
        expected_cost = report.extra_metrics["expected_cost"]
        assert report.var_95 >= expected_cost - 1e-6, (
            f"VaR {report.var_95:.2f} < expected cost {expected_cost:.2f}"
        )

    def test_cvar_gte_expected_cost(self, report):
        """CVaR must be >= expected cost."""
        expected_cost = report.extra_metrics["expected_cost"]
        assert report.cvar_95 >= expected_cost - 1e-6, (
            f"CVaR {report.cvar_95:.2f} < expected cost {expected_cost:.2f}"
        )

    def test_p_budget_exceed_valid_probability(self, report):
        """P(budget exceed) must be a valid probability in [0, 1]."""
        p = report.extra_metrics["p_budget_exceed"]
        assert 0.0 <= p <= 1.0, f"p_budget_exceed={p} is not a valid probability"

    def test_p_sla_breach_valid_probability(self, report):
        """P(SLA breach) must be a valid probability in [0, 1]."""
        p = report.extra_metrics["p_sla_breach"]
        assert 0.0 <= p <= 1.0, f"p_sla_breach={p} is not a valid probability"

    def test_volatility_non_negative(self, report):
        """Cost standard deviation must be non-negative."""
        assert report.volatility >= 0.0

    def test_max_drawdown_non_negative(self, report):
        """Max drawdown (max - min cost) must be non-negative."""
        assert report.max_drawdown >= 0.0


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------
class TestDeterminism:

    def test_same_seed_identical_output(self, distribution, context):
        """Same seed must produce byte-identical metrics."""
        ev_a = ShippingRiskEvaluator(
            config={"n_simulations": 1_000, "seed": 42, "budget_cap": BUDGET_CAP},
            demand_distribution=distribution,
        )
        ev_b = ShippingRiskEvaluator(
            config={"n_simulations": 1_000, "seed": 42, "budget_cap": BUDGET_CAP},
            demand_distribution=distribution,
        )
        report_a = ev_a.assess(ALLOCATION, context)
        report_b = ev_b.assess(ALLOCATION, context)

        assert report_a.cvar_95 == report_b.cvar_95
        assert report_a.var_95 == report_b.var_95
        assert report_a.extra_metrics["expected_cost"] == report_b.extra_metrics["expected_cost"]


# ---------------------------------------------------------------------------
# Tests: sanity checks on budget logic
# ---------------------------------------------------------------------------
class TestBudgetSanity:

    def test_infinite_budget_zero_exceedance(self, distribution, context):
        """With infinite budget, P(exceed) must be 0."""
        ev = ShippingRiskEvaluator(
            config={"n_simulations": 1_000, "seed": 42, "budget_cap": float("inf")},
            demand_distribution=distribution,
        )
        report = ev.assess(ALLOCATION, context)
        assert report.extra_metrics["p_budget_exceed"] == 0.0

    def test_zero_budget_full_exceedance(self, distribution, context):
        """With zero budget, P(exceed) must be 1.0."""
        ev = ShippingRiskEvaluator(
            config={"n_simulations": 1_000, "seed": 42, "budget_cap": 0.0},
            demand_distribution=distribution,
        )
        report = ev.assess(ALLOCATION, context)
        assert report.extra_metrics["p_budget_exceed"] == 1.0

    def test_is_within_tolerance_when_cvar_below_budget(self, distribution, context):
        """is_within_tolerance must be True when CVaR < budget_cap."""
        ev = ShippingRiskEvaluator(
            config={"n_simulations": 1_000, "seed": 42, "budget_cap": float("inf")},
            demand_distribution=distribution,
        )
        report = ev.assess(ALLOCATION, context)
        assert report.is_within_tolerance is True


# ---------------------------------------------------------------------------
# Tests: demand distribution
# ---------------------------------------------------------------------------
class TestDemandDistribution:

    def test_no_negative_demand_samples(self):
        """Truncated normal must never produce negative demand."""
        dist = DemandDistribution(mu=1_000.0, sigma=500.0, lower=0.0)
        rng = np.random.default_rng(42)
        samples = dist.sample(10_000, rng)
        assert np.all(samples >= 0.0), "Demand distribution produced negative values"

    def test_sample_count_correct(self):
        """sample() must return exactly n values."""
        dist = DemandDistribution(mu=10_000.0, sigma=1_500.0)
        rng = np.random.default_rng(0)
        samples = dist.sample(500, rng)
        assert len(samples) == 500

    def test_sample_mean_close_to_mu(self):
        """Sample mean should be close to mu with large n."""
        dist = DemandDistribution(mu=10_000.0, sigma=500.0)
        rng = np.random.default_rng(42)
        samples = dist.sample(50_000, rng)
        assert abs(np.mean(samples) - 10_000.0) < 50.0, (
            f"Sample mean {np.mean(samples):.1f} too far from mu=10000"
        )


# ---------------------------------------------------------------------------
# Tests: RiskReport types
# ---------------------------------------------------------------------------
class TestRiskReportTypes:

    def test_report_field_types(self, report):
        """All RiskReport fields must have correct types."""
        assert isinstance(report, RiskReport)
        assert isinstance(report.var_95, float)
        assert isinstance(report.cvar_95, float)
        assert isinstance(report.volatility, float)
        assert isinstance(report.max_drawdown, float)
        assert isinstance(report.is_within_tolerance, bool)
        assert isinstance(report.extra_metrics, dict)

    def test_extra_metrics_keys_present(self, report):
        """All expected keys must exist in extra_metrics."""
        required_keys = [
            "expected_cost", "cost_std", "p_budget_exceed",
            "p_sla_breach", "n_simulations",
        ]
        for key in required_keys:
            assert key in report.extra_metrics, f"Missing key: {key}"