"""Observability tests (plan §7/§9): /metrics coverage, alert↔metric mapping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import yaml
from fastapi.testclient import TestClient

import bis_assistant.server as srv
from bis_assistant import metrics as metrics_mod

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SERIES = ["bis_chat_total", "bis_answered_total", "bis_refused_total",
                   "bis_needs_info_total", "bis_model_unavailable_total",
                   "bis_chat_latency_p50_ms",
                   "bis_chat_latency_p95_ms", "bis_citation_fail_total",
                   "bis_kb_staleness_days", "bis_feedback_total",
                   "bis_feedback_neg_total", "bis_chat_5xx_total"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "DB_PATH", tmp_path / "ops.db")
    metrics_mod.reset()
    with TestClient(srv.app) as c:
        yield c


def test_metrics_covers_plan_series(client):
    client.post("/chat", json={"query": "steel bottle"})
    body = client.get("/metrics").text
    for series in REQUIRED_SERIES:
        assert series in body, series


def test_model_unavailability_is_not_counted_as_an_answer(client):
    client.post("/chat", json={"query": "steel bottle"})
    client.post("/chat", json={"query": "guarantee my licence"})
    snap = metrics_mod.snapshot()
    assert snap["chat_total"] >= 2
    assert snap["model_unavailable_total"] >= 2
    assert snap["answered_total"] == 0 and snap["chat_latency_p95_ms"] >= 0


def test_verifier_trip_surfaces(client):
    from bis_assistant import verifier
    before = metrics_mod.snapshot()["citation_fail_total"]
    verifier.verify_grounded_response("Use IS 1234 [Source 1].", [])
    assert metrics_mod.snapshot()["citation_fail_total"] == before + 1


def test_alerts_map_to_metrics():
    rules = yaml.safe_load((ROOT / "ops" / "alerts.yaml").read_text())["alerts"]
    assert len(rules) >= 5
    for rule in rules:
        assert rule.get("playbook", "").startswith("runbook#"), rule["name"]
        assert "bis_" in rule["expr"], rule["name"]
