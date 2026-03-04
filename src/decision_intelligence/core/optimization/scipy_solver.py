from __future__ import annotations
from typing import Any, Dict, List
import itertools
import numpy as np
from scipy.optimize import minimize

from decision_intelligence.core.optimization.base import (
    BaseSolver,
    OptimizationResult,
    SearchSpace,
)
from decision_intelligence.core.objective.base import BaseObjective
from decision_intelligence.core.constraints.base import ConstraintRegistry


class MixedScipySolver(BaseSolver):
    """
    Mixed discrete + continuous solver.

    Strategy:
    1. Enumerate discrete combinations.
    2. For each combination, solve continuous sub-problem via SLSQP.
    3. Apply penalties + feasibility checks.
    4. Return best feasible solution.

    Why SLSQP (not L-BFGS-B):
        L-BFGS-B handles bounds only.
        SLSQP handles bounds + equality + inequality constraints.
        Shipping allocation requires Σ x_i = 1 (equality) and
        budget cap (inequality) — both impossible with L-BFGS-B.

    Config keys (passed at construction):
        n_restarts : int  — number of random starting points (default 5)
        seed       : int  — global RNG seed for reproducibility (default 42)
    """

    def solve(
        self,
        objective: BaseObjective,
        constraints: ConstraintRegistry,
        search_space: SearchSpace,
        context: Dict[str, Any],
    ) -> OptimizationResult:

        bounds = search_space.bounds
        discrete_vars = search_space.discrete_vars
        metadata = search_space.metadata

        n_restarts: int = self.config.get("n_restarts", 5)
        seed: int = self.config.get("seed", 42)
        rng = np.random.default_rng(seed)

        continuous_vars: List[str] = [
            k for k in bounds.keys() if k not in discrete_vars
        ]

        # ── Discrete combinations ─────────────────────────────────────────
        discrete_grids = {
            var: np.arange(bounds[var][0], bounds[var][1] + 1)
            for var in discrete_vars
        }
        discrete_combinations = (
            [{}]
            if not discrete_grids
            else [
                dict(zip(discrete_grids.keys(), values))
                for values in itertools.product(*discrete_grids.values())
            ]
        )

        best_solution = None
        best_value = np.inf
        best_violations = []
        total_iterations = 0

        # ── SLSQP constraints ─────────────────────────────────────────────
        # Built once — same for every discrete combination and restart.
        slsqp_constraints = []

        # Equality: Σ x_i = 1 (full demand allocation, no free lunch)
        if continuous_vars:
            slsqp_constraints.append({
                "type": "eq",
                "fun": lambda x: np.sum(x) - 1.0,
                "jac": lambda x: np.ones(len(x)),
            })

        # Inequality: budget_cap - total_cost >= 0 (hard budget constraint)
        # Coefficients pulled from search_space.metadata — business concern,
        # not solver concern. Solver stays domain-agnostic.
        budget_cap: float | None = metadata.get("budget_cap")
        demand: float | None = metadata.get("demand")
        cost_coeffs: list | None = metadata.get("cost_coeffs")  # cost per unit per vendor

        if budget_cap is not None and demand is not None and cost_coeffs is not None:
            coeffs = np.array(cost_coeffs) * demand
            slsqp_constraints.append({
                "type": "ineq",
                # scipy ineq: fun(x) >= 0
                # budget_cap - Σ(x_i * cost_i * demand) >= 0
                "fun": lambda x, c=coeffs, b=budget_cap: b - np.dot(x, c),
                "jac": lambda x, c=coeffs: -c,
            })

        # ── Solve ─────────────────────────────────────────────────────────
        for discrete_choice in discrete_combinations:

            cont_bounds = [bounds[v] for v in continuous_vars]

            def wrapped_objective(x: np.ndarray, dc: dict = discrete_choice) -> float:
                solution = {**dc, **dict(zip(continuous_vars, x))}
                result = objective.evaluate(solution, context)
                penalty = constraints.total_penalty(solution, context)
                return objective.sign() * (result.value + penalty)

            if continuous_vars:
                for restart in range(n_restarts):
                    # Random start on probability simplex (Dirichlet)
                    x0 = rng.dirichlet(np.ones(len(continuous_vars)))
                    # Clip to bounds and renormalize
                    lo = np.array([b[0] for b in cont_bounds])
                    hi = np.array([b[1] for b in cont_bounds])
                    x0 = np.clip(x0, lo, hi)
                    x0 = x0 / x0.sum() if x0.sum() > 0 else np.ones(len(continuous_vars)) / len(continuous_vars)

                    res = minimize(
                        wrapped_objective,
                        x0,
                        method="SLSQP",
                        bounds=cont_bounds,
                        constraints=slsqp_constraints,
                        options={"maxiter": 1000, "ftol": 1e-9},
                    )
                    total_iterations += res.nit

                    if not res.success:
                        continue

                    solution = {**discrete_choice, **dict(zip(continuous_vars, res.x))}
                    violations = constraints.evaluate_all(solution, context)

                    if constraints.is_feasible(solution, context) and res.fun < best_value:
                        best_value = res.fun
                        best_solution = solution
                        best_violations = violations

            else:
                solution = discrete_choice
                result = objective.evaluate(solution, context)
                value = objective.sign() * result.value
                violations = constraints.evaluate_all(solution, context)
                total_iterations += 1

                if constraints.is_feasible(solution, context) and value < best_value:
                    best_value = value
                    best_solution = solution
                    best_violations = violations

        return OptimizationResult(
            best_solution=best_solution,
            best_value=float(best_value) if best_solution is not None else float("inf"),
            converged=best_solution is not None,
            iterations=total_iterations,
            is_feasible=best_solution is not None,
            constraint_violations=best_violations,
        )