"""Tests for governance.audit"""
from decision_intelligence.governance.audit import AuditLogger, AuditRecord


def test_audit_logger_stores_records():
    logger = AuditLogger()
    record = AuditRecord(run_id="abc", timestamp="2024-01-01T00:00:00Z", component="test")
    logger.log(record)
    assert len(logger.records()) == 1
    assert logger.records()[0].run_id == "abc"


def test_audit_logger_is_immutable_copy():
    logger = AuditLogger()
    logger.log(AuditRecord(run_id="x", timestamp="t", component="c"))
    copy = logger.records()
    copy.clear()
    assert len(logger.records()) == 1
