"""Lightweight in-process model registry."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}

    def register(self, name: str, model: Any, metadata: Optional[Dict] = None) -> None:
        self._models[name] = {"model": model, "metadata": metadata or {}}

    def get(self, name: str) -> Any:
        return self._models[name]["model"]

    def list_models(self) -> List[str]:
        return list(self._models.keys())
