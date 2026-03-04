"""Simulation pipeline — Monte-Carlo forward simulation."""
from __future__ import annotations
from typing import Any, Dict


class SimulatePipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self) -> None:
        raise NotImplementedError
