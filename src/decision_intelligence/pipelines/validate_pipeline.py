"""Validation pipeline — back-tests decisions against historical data."""
from __future__ import annotations
from typing import Any, Dict


class ValidatePipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self) -> None:
        raise NotImplementedError
