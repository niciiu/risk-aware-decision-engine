"""Tests for core.constraints.base"""
from decision_intelligence.core.constraints.base import ConstraintRegistry


def test_empty_registry_is_feasible():
    r = ConstraintRegistry()
    assert r.is_feasible(solution=None, context={}) is True


def test_empty_registry_zero_penalty():
    r = ConstraintRegistry()
    assert r.total_penalty(solution=None, context={}) == 0.0
