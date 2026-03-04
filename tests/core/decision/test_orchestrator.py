"""Smoke tests for DecisionOrchestrator (no real engine needed)."""


def test_import():
    from decision_intelligence.core.decision.orchestrator import (
        DecisionOrchestrator,
        OrchestratorResult,
    )
    assert DecisionOrchestrator is not None
    assert OrchestratorResult is not None
