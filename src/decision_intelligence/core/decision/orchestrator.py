"""
DecisionOrchestrator — the full decision loop in one place.

Wires every sub-system together in an explicit, auditable pipeline:

    1. Forecast          — probabilistic signal generation
    2. Regime detect     — context conditioning
    3. Candidate actions — domain-specific action space (via engine)
    4. Objective eval    — score candidates
    5. Constraint check  — prune infeasible candidates
    6. Solver            — global search over feasible space
    7. Risk assessment   — post-hoc safety gate
    8. Audit log         — immutable trace of every run

Design goals
------------
- Every step is replaceable via dependency injection.
- The orchestrator contains NO domain logic.
- Failures are loud — no silent fallbacks that hide bugs in production.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from decision_intelligence.core.decision.base import (
    BaseDecisionEngine,
    DecisionContext,
    DecisionOutput,
)
from decision_intelligence.core.forecasting.base import BaseForecaster, ForecastResult
from decision_intelligence.core.regime.base import BaseRegimeDetector, RegimeState
from decision_intelligence.core.risk.base import RiskReport
from decision_intelligence.core.optimization.base import OptimizationResult
from decision_intelligence.governance.audit import AuditLogger, AuditRecord


@dataclass
class OrchestratorResult:
    """Full trace of a single orchestrator run."""
    run_id: str
    timestamp: str
    decision: DecisionOutput
    forecast: Optional[ForecastResult]
    regime: Optional[RegimeState]
    optimization: Optional[OptimizationResult]
    risk: Optional[RiskReport]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionOrchestrator:
    """
    Wires:
        Forecaster -> RegimeDetector -> DecisionEngine
        (Objective + ConstraintRegistry + Solver) -> RiskModel -> AuditLogger

    into a single reproducible decision loop.

    Parameters
    ----------
    engine           : fully-configured BaseDecisionEngine
    forecaster       : optional — for forecast-conditioned decisions
    regime_detector  : optional — for regime-aware objectives
    audit_logger     : optional — for immutable audit trails
    """

    def __init__(
        self,
        engine: BaseDecisionEngine,
        forecaster: Optional[BaseForecaster] = None,
        regime_detector: Optional[BaseRegimeDetector] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self.engine = engine
        self.forecaster = forecaster
        self.regime_detector = regime_detector
        self.audit_logger = audit_logger

    # ──────────────────────────────────────────────────────────────── public

    def run(
        self,
        raw_features: Dict[str, Any],
        domain: str,
        forecast_input: Optional[Any] = None,
        regime_input: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """
        Execute the full 8-step decision loop.

        Parameters
        ----------
        raw_features   : domain-specific feature dict
        domain         : domain tag, e.g. "shipping"
        forecast_input : X passed to forecaster.predict()
        regime_input   : X passed to regime_detector.detect()
        extra_metadata : arbitrary key/values appended to the audit record
        """
        run_id    = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Forecast ─────────────────────────────────────────────────
        forecast: Optional[ForecastResult] = None
        if self.forecaster is not None and forecast_input is not None:
            forecast = self.forecaster.predict(forecast_input)
            raw_features["forecast_point"]       = forecast.point
            raw_features["forecast_uncertainty"] = forecast.uncertainty

        # ── Step 2: Regime detection ─────────────────────────────────────────
        regime: Optional[RegimeState] = None
        if self.regime_detector is not None and regime_input is not None:
            regime = self.regime_detector.dominant_regime(regime_input)
            raw_features["regime_label"]       = regime.label
            raw_features["regime_probability"] = regime.probability

        # ── Steps 3–6: Decision engine ───────────────────────────────────────
        # (action space → objective → constraints → solver)
        context = DecisionContext(
            timestamp=timestamp,
            domain=domain,
            features=raw_features,
            forecast=forecast,
            regime=regime,
            metadata=extra_metadata or {},
        )
        decision: DecisionOutput = self.engine.decide(context)

        # ── Step 7: Risk gate ─────────────────────────────────────────────────
        risk: Optional[RiskReport] = decision.risk_report
        if risk is not None and not risk.is_within_tolerance:
            decision.metadata["risk_gate_failed"] = True

        # ── Step 8: Audit log ─────────────────────────────────────────────────
        result = OrchestratorResult(
            run_id=run_id,
            timestamp=timestamp,
            decision=decision,
            forecast=forecast,
            regime=regime,
            optimization=decision.optimization_result,
            risk=risk,
            metadata=extra_metadata or {},
        )
        self._audit(run_id, timestamp, domain, raw_features, result)
        return result

    # ──────────────────────────────────────────────────────────────── private

    def _audit(
        self,
        run_id: str,
        timestamp: str,
        domain: str,
        inputs: Dict[str, Any],
        result: OrchestratorResult,
    ) -> None:
        if self.audit_logger is None:
            return
        self.audit_logger.log(AuditRecord(
            run_id=run_id,
            timestamp=timestamp,
            component="DecisionOrchestrator",
            inputs={"domain": domain, "features_keys": list(inputs.keys())},
            outputs={
                "action":    result.decision.action,
                "confidence": result.decision.confidence,
                "converged": (
                    result.optimization.converged if result.optimization else None
                ),
            },
            tags={"domain": domain},
        ))
