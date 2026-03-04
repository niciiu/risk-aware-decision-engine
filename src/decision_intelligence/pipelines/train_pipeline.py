"""Training pipeline — fits forecasters and regime detectors."""
from __future__ import annotations
from typing import Any, Dict


class TrainPipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self) -> None:
        raise NotImplementedError
