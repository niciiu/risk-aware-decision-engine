"""Explainability abstractions."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, model: Any, input_data: Any) -> Dict[str, Any]: ...


class PassthroughExplainer(BaseExplainer):
    def explain(self, model: Any, input_data: Any) -> Dict[str, Any]:
        return {"explanation": str(model), "input": str(input_data)}
