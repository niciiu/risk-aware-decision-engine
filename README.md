# Risk-Aware Decision Engine

A production-oriented framework for **operational decision-making under uncertainty**.

Built around the observation that real procurement decisions are not single-metric problems — they involve competing objectives, hard operational constraints, and input data that is fundamentally uncertain. This project implements a decision engine that handles all three explicitly, using vendor allocation in shipping logistics as the reference domain.

---

## Motivation

Most optimization tools optimize a single metric. In practice, supply chain decisions require balancing cost, service level, and vendor reliability simultaneously — with a budget that cannot be exceeded and SLA commitments that carry real penalties.

The goal here was to build a system that:
- makes the tradeoffs explicit and configurable by business stakeholders
- quantifies how bad decisions get under demand uncertainty (not just expected case)
- produces an auditable reasoning trace for every allocation decision
- is structured so the core framework can be reused across domains beyond shipping

---

## Quick Start

```bash
git clone https://github.com/niciiu/risk-aware-decision-engine.git
cd risk-aware-decision-engine
python -m venv decision_venv && decision_venv\Scripts\activate   # Windows
pip install -e ".[dev]"
pytest

python examples/run_phase1.py   # deterministic allocation
python examples/run_phase2.py   # probabilistic risk evaluation
python examples/run_phase3.py   # comparison against heuristic strategies
```

---

## System Architecture

The framework separates domain logic from optimization mechanics. The solver knows nothing about shipping — it receives an objective function, a constraint registry, and a search space. The domain adapter translates business inputs into those abstractions.

```
src/decision_intelligence/
├── core/
│   ├── decision/           # BaseDecisionEngine + 8-step orchestrator
│   ├── objective/          # BaseObjective, ObjectiveResult, CompositeObjective
│   ├── constraints/        # Hard/soft constraint registry with penalty scoring
│   ├── optimization/       # MixedScipySolver — SLSQP with multi-restart
│   ├── risk/               # MonteCarloRiskEvaluator — VaR-95, CVaR-95
│   ├── forecasting/        # Demand forecast abstractions (quantiles + samples)
│   └── regime/             # Regime detection interface (Phase 4 extension point)
├── domain/
│   └── shipping/           # Concrete shipping objective, constraints, engine, adapter
└── governance/
    ├── audit.py            # Immutable decision audit log
    ├── explainability.py   # Structured decision reasoning and weight sensitivity
    └── model_registry.py   # In-process model registry
```

The `core/` layer has no shipping-specific logic. Swapping in a different domain (e.g. procurement, capacity planning) requires implementing the domain adapter only.

---

## Decision Orchestration

Every allocation decision passes through an 8-step pipeline:

```
raw_features → [1] Forecast → [2] Regime Detect → [3] Build SearchSpace
             → [4] Evaluate Objective → [5] Check Constraints
             → [6] Solve → [7] Risk Gate → [8] Audit Log
             → OrchestratorResult
```

Steps 1–2 condition the search space on current market state. Steps 4–6 run the optimization. Step 7 evaluates CVaR on the resulting allocation before the decision is committed. Step 8 produces an immutable audit record regardless of outcome.

---

## Optimization

**Solver**: `MixedScipySolver` — supports discrete + continuous variable spaces via SLSQP.

SLSQP was chosen over L-BFGS-B because the allocation problem requires both an equality constraint (`Σ x_i = 1`) and a budget inequality constraint. L-BFGS-B handles bounds only.

Multi-restart with Dirichlet initialization reduces sensitivity to starting point on non-convex objective landscapes. Output is fully deterministic under a fixed seed.

**Objective function** (shipping domain):

```
f(x) = w_cost × Σ(x_i · cost_i · demand)
     + w_sla  × Σ(x_i · max(0, lead_i − threshold) · penalty · demand)
     + w_risk × Σ(x_i · (1 − reliability_i))
```

Weights are externalized to `configs/` — business teams can adjust the cost/SLA/risk tradeoff without touching code.

---

## Uncertainty Quantification (Phase 2)

Demand is modeled as a truncated Normal distribution (`μ = 10,000`, `σ = 1,500`, `lower = 0`) to avoid physically meaningless negative samples. 10,000 Monte Carlo scenarios are evaluated against the optimal allocation.

| Metric | Value |
|---|---|
| Expected cost | $39,539 |
| Std deviation | $5,932 |
| VaR-95 | $49,216 |
| CVaR-95 | $51,905 |
| P(exceed $55k budget) | 0.5% |

CVaR was chosen over VaR as the primary risk metric because it captures the magnitude of tail losses, not just their threshold. A 0.5% budget exceedance probability with CVaR of $51,905 against a $55,000 cap means the allocation is robust under realistic demand variability.

---

## Baseline Comparison (Phase 3)

Three heuristic strategies were evaluated against the engine on `effective_cost = shipping_cost + SLA_penalty`:

| Strategy | Effective Cost | Notes |
|---|---|---|
| **Engine** | **$39,600** | Optimized cost-SLA-risk tradeoff |
| Balanced SLA | $44,000 | Moderate cost, avoids SLA breach |
| Highest reliability | $45,600 | Lowest risk, highest cost |
| Cheapest first | $47,750 | Lowest raw cost, full SLA penalty |

The cheapest-first heuristic concentrates volume on the lowest-cost vendor, which violates the SLA threshold and incurs $11,250 in penalties — more than offsetting the cost advantage. The engine avoids this entirely.

Effective cost improvement over best heuristic: **10%**. Risk exposure improvement: **44%**.

---

## Governance

### Audit Trail

Every decision — including infeasible outcomes — produces an immutable `AuditRecord` capturing the full input context, optimization result, constraint status, and risk metrics. The audit log is queryable after each run.

### Decision Explainability

`DecisionExplainer` translates solver output into structured reasoning that non-technical stakeholders can read and sign off on. Output includes:

- **Vendor rationale**: why each vendor was included or excluded, referencing SLA, cost rank, and reliability
- **Objective decomposition**: which term (cost / SLA / risk) dominated the weighted objective
- **Constraint status**: budget, SLA, and allocation integrity with slack values
- **Weight sensitivity**: first-order partial derivatives `∂objective/∂weight`, identifying which business levers have the most influence on the decision

```python
explainer = DecisionExplainer(
    vendor_profiles=vendors,
    w_cost=0.60, w_sla=0.25, w_risk=0.15,
    sla_threshold_days=3.0,
    penalty_per_day=1.5,
    budget_cap=55_000.0,
)
explanation = explainer.explain(solution, demand=10_000.0, decision_id="run_001")
print(explanation.to_text())
audit_logger.log(..., outputs=explanation.to_dict())
```

---

## Test Coverage

Tests are organized to mirror the source structure. Critical paths — solver correctness, risk metric ordering, and constraint enforcement — are covered explicitly.

```
tests/
├── core/
│   ├── constraints/    test_base.py           constraint registry and penalty computation
│   ├── decision/       test_orchestrator.py   end-to-end decision loop integration
│   ├── forecasting/    test_base.py           forecast result contracts
│   ├── optimization/   test_scipy_solver.py   solver feasibility, determinism, infeasibility handling
│   └── risk/           test_monte_carlo.py    CVaR/VaR ordering, budget exceedance logic, distribution sampling
└── governance/         test_audit.py          audit record integrity
```

Key invariants verified:

- Allocation sum equals 1.0 within numerical tolerance (equality constraint)
- Budget constraint is never violated on feasible problems
- CVaR ≥ VaR ≥ E[cost] holds across all simulation seeds
- Solver returns `converged=False` on provably infeasible problems
- Identical allocation under identical seed (reproducibility)

Run:
```bash
pytest -v
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Complete | Deterministic multi-objective optimization with constraint registry |
| 2 | ✅ Complete | Monte Carlo uncertainty modeling with CVaR risk gate |
| 3 | ✅ Complete | Heuristic baseline benchmarking |
| 4 | 🔄 Planned | Historical backtesting on synthetic operational data |
| 5 | 🔄 Planned | Decision regret analysis and full weight sensitivity surface |

---

## Configuration

All scenario parameters live in `configs/`. No business logic is hardcoded.

```yaml
# configs/phase1_baseline.yaml (excerpt)
objective_weights:
  w_cost: 0.60
  w_sla:  0.25
  w_risk: 0.15

constraints:
  budget_cap:
    value: 55000
  sla_threshold:
    threshold_days: 3.0
    penalty_per_day: 1.5
```

---

## Dependencies

- `scipy` — SLSQP optimization
- `numpy` — vectorized objective and risk computation
- `pyyaml` — config loading
- `pytest` — test runner