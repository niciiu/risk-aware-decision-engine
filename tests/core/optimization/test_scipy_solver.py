"""
Unit tests for MixedScipySolver.

What we test and why:
    1. allocation sums to 1.0       — equality constraint must always hold
    2. all allocations within bounds — solver must respect per-vendor bounds
    3. budget constraint respected   — hard constraint must not be violated
    4. result is feasible            — converged=True, is_feasible=True
    5. deterministic output          — same seed → same result always
    6. infeasible problem detected   — solver returns converged=False gracefully
"""
from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.core.optimization.scipy_solver import MixedScipySolver
from decision_intelligence.core.optimization.base import SearchSpace, OptimizationResult
from decision_intelligence.core.objective.base import BaseObjective, ObjectiveResult
from decision_intelligence.core.constraints.base import ConstraintRegistry


# ---------------------------------------------------------------------------
# Minimal concrete objective for testing
# Minimise weighted cost: objective = sum(cost_coeffs * allocation * demand)
# ---------------------------------------------------------------------------
class SimpleCostObjective(BaseObjective):
    """Minimise total shipping cost."""

    def evaluate(self, solution: dict, context: dict) -> ObjectiveResult:
        demand = context["demand"]
        cost_coeffs = context["cost_coeffs"]
        vendors = context["vendors"]
        cost = sum(
            solution[v] * cost_coeffs[i] * demand
            for i, v in enumerate(vendors)
        )
        return ObjectiveResult(value=cost, breakdown={"cost": cost})

    def sign(self) -> float:
        return 1.0  # minimise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
VENDORS = ["vendor_A", "vendor_B", "vendor_C"]
COST_COEFFS = [3.5, 4.0, 4.5]   # cost per unit per vendor
DEMAND = 10_000.0
BUDGET_CAP = 50_000.0            # comfortably above minimum possible cost


@pytest.fixture
def context():
    return {
        "demand": DEMAND,
        "cost_coeffs": COST_COEFFS,
        "vendors": VENDORS,
    }


@pytest.fixture
def search_space():
    return SearchSpace(
        bounds={v: (0.0, 1.0) for v in VENDORS},
        discrete_vars=[],
        metadata={
            "budget_cap": BUDGET_CAP,
            "demand": DEMAND,
            "cost_coeffs": COST_COEFFS,
        },
    )


@pytest.fixture
def objective():
    return SimpleCostObjective(config={})


@pytest.fixture
def constraint_registry():
    return ConstraintRegistry()   # no extra constraints — budget handled in solver


@pytest.fixture
def solver():
    return MixedScipySolver(config={"n_restarts": 3, "seed": 42})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMixedScipySolverBasic:

    def test_allocation_sums_to_one(self, solver, objective, constraint_registry, search_space, context):
        """Equality constraint: Σ x_i = 1 must always hold."""
        result = solver.solve(objective, constraint_registry, search_space, context)

        assert result.converged, "Solver should converge on a feasible problem"
        total = sum(result.best_solution[v] for v in VENDORS)
        assert abs(total - 1.0) < 1e-6, f"Allocation sum = {total}, expected 1.0"

    def test_all_allocations_within_bounds(self, solver, objective, constraint_registry, search_space, context):
        """Each vendor allocation must stay within [0, 1]."""
        result = solver.solve(objective, constraint_registry, search_space, context)

        for v in VENDORS:
            alloc = result.best_solution[v]
            assert 0.0 - 1e-6 <= alloc <= 1.0 + 1e-6, (
                f"{v} allocation {alloc:.4f} out of bounds [0, 1]"
            )

    def test_budget_constraint_respected(self, solver, objective, constraint_registry, search_space, context):
        """Hard budget constraint: total cost must not exceed budget_cap."""
        result = solver.solve(objective, constraint_registry, search_space, context)

        total_cost = sum(
            result.best_solution[v] * COST_COEFFS[i] * DEMAND
            for i, v in enumerate(VENDORS)
        )
        assert total_cost <= BUDGET_CAP + 1e-3, (
            f"Total cost {total_cost:.2f} exceeds budget cap {BUDGET_CAP}"
        )

    def test_result_is_feasible(self, solver, objective, constraint_registry, search_space, context):
        """Solver must mark result as feasible on a feasible problem."""
        result = solver.solve(objective, constraint_registry, search_space, context)

        assert result.is_feasible is True
        assert result.converged is True

    def test_result_has_valid_types(self, solver, objective, constraint_registry, search_space, context):
        """OptimizationResult fields must have correct types."""
        result = solver.solve(objective, constraint_registry, search_space, context)

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_value, float)
        assert isinstance(result.iterations, int)
        assert result.iterations > 0


class TestMixedScipySolverDeterminism:

    def test_same_seed_produces_same_result(self, objective, constraint_registry, search_space, context):
        """Reproducibility: same seed must always produce identical output."""
        solver_a = MixedScipySolver(config={"n_restarts": 3, "seed": 42})
        solver_b = MixedScipySolver(config={"n_restarts": 3, "seed": 42})

        result_a = solver_a.solve(objective, constraint_registry, search_space, context)
        result_b = solver_b.solve(objective, constraint_registry, search_space, context)

        assert abs(result_a.best_value - result_b.best_value) < 1e-9
        for v in VENDORS:
            assert abs(result_a.best_solution[v] - result_b.best_solution[v]) < 1e-9

    def test_different_seed_may_differ(self, objective, constraint_registry, search_space, context):
        """Different seeds are allowed to find different (but both valid) solutions."""
        solver_42 = MixedScipySolver(config={"n_restarts": 3, "seed": 42})
        solver_99 = MixedScipySolver(config={"n_restarts": 3, "seed": 99})

        result_42 = solver_42.solve(objective, constraint_registry, search_space, context)
        result_99 = solver_99.solve(objective, constraint_registry, search_space, context)

        # Both must still be feasible — regardless of seed
        assert result_42.is_feasible is True
        assert result_99.is_feasible is True


class TestMixedScipySolverOptimality:

    def test_minimises_cost(self, solver, objective, constraint_registry, search_space, context):
        """
        Cheapest vendor (vendor_A, cost=3.5) should receive highest allocation.
        Under pure cost minimisation with no other constraints,
        optimal solution concentrates on cheapest vendor.
        """
        result = solver.solve(objective, constraint_registry, search_space, context)

        alloc_A = result.best_solution["vendor_A"]
        alloc_C = result.best_solution["vendor_C"]

        assert alloc_A >= alloc_C - 1e-4, (
            f"Expected vendor_A (cheapest) >= vendor_C (most expensive), "
            f"got A={alloc_A:.4f}, C={alloc_C:.4f}"
        )

    def test_infeasible_budget_returns_not_converged(self, objective, constraint_registry, context):
        """
        If budget cap is impossibly tight, solver should return converged=False.
        Minimum possible cost = cheapest_vendor * demand = 3.5 * 10000 = 35000.
        Setting budget_cap=1000 makes the problem infeasible.
        """
        tight_space = SearchSpace(
            bounds={v: (0.0, 1.0) for v in VENDORS},
            discrete_vars=[],
            metadata={
                "budget_cap": 1_000.0,   # impossibly tight
                "demand": DEMAND,
                "cost_coeffs": COST_COEFFS,
            },
        )
        solver = MixedScipySolver(config={"n_restarts": 3, "seed": 42})
        result = solver.solve(objective, constraint_registry, tight_space, context)

        assert result.converged is False, (
            "Solver should not converge on an infeasible problem"
        )
        assert result.is_feasible is False