# Decision Intelligence Framework

Domain-agnostic, production-grade framework for **uncertainty-aware optimization under constraints**.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python main.py --pipeline train --config configs/global.yaml
```

## Architecture

```
src/decision_intelligence/
├── core/
│   ├── forecasting/        # ForecastResult (quantiles + samples, not just lower/upper)
│   ├── decision/
│   │   ├── base.py         # DecisionEngine owns Objective + Constraints + RiskModel + Solver
│   │   └── orchestrator.py # Full 8-step decision loop
│   ├── constraints/        # Hard/soft constraints + ConstraintRegistry
│   ├── regime/             # Regime detection
│   ├── objective/          # BaseObjective + CompositeObjective
│   ├── risk/               # VaR / CVaR / drawdown
│   └── optimization/       # BaseSolver(objective, constraints, search_space, context)
├── domain/
│   └── shipping/           # Shipping domain adapter
├── governance/             # Audit logging, model registry, explainability
└── pipelines/              # Train / Validate / Simulate
```

## Decision Loop (`orchestrator.py`)

```
raw_features
     │
     ▼
[1] Forecast        → ForecastResult  (quantiles + samples)
     │
     ▼
[2] Regime Detect   → RegimeState     (label + probability)
     │
     ▼
[3] SearchSpace     → variable bounds (domain-specific)
     │
     ▼
[4] Objective eval  → ObjectiveResult (value + components)
     │
     ▼
[5] Constraints     → feasibility gate + soft penalties
     │
     ▼
[6] Solver.solve()  → OptimizationResult (best_solution)
     │
     ▼
[7] Risk gate       → RiskReport (VaR, CVaR, drawdown)
     │
     ▼
[8] Audit log       → AuditRecord (immutable trace)
     │
     ▼
OrchestratorResult
```
