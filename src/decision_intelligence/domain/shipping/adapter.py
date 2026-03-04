"""
Shipping domain adapter.

Translates raw shipping data -> DecisionContext and
OptimizationResult.best_solution -> a human-readable carrier action.
"""
from __future__ import annotations
from typing import Any, Dict

from decision_intelligence.core.decision.base import BaseDecisionEngine, DecisionContext
from decision_intelligence.core.optimization.base import SearchSpace


class ShippingDecisionAdapter(BaseDecisionEngine):
    """
    Shipping-domain concrete engine.
    Implements the two domain-specific hooks only — no optimization logic.
    """

    def build_search_space(self, context: DecisionContext) -> SearchSpace:
        """Variable space for shipping optimisation — override with real bounds."""
        raise NotImplementedError

    def solution_to_action(self, solution: Any, context: DecisionContext) -> str:
        """Map optimiser output (e.g. carrier index) to a readable action string."""
        raise NotImplementedError


class ShippingFeatureBuilder:
    """Converts raw shipping API / DB rows into a feature dict."""

    def build(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
