"""Immutable audit logging."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AuditRecord:
    run_id: str
    timestamp: str
    component: str
    inputs: Dict[str, Any]  = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str]    = field(default_factory=dict)


class AuditLogger:
    def __init__(self) -> None:
        self._records: List[AuditRecord] = []

    def log(self, record: AuditRecord) -> None:
        self._records.append(record)

    def records(self) -> List[AuditRecord]:
        return list(self._records)
