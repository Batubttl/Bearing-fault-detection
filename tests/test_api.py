import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bearing_fault_detection.api import app

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # "with" runs the startup code -> model is loaded
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_valid_window(client):
    body = json.loads((SAMPLES / "inner_race.json").read_text())
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in data["probabilities"]
    assert abs(sum(data["probabilities"].values()) - 1) < 1e-3


def test_predict_rejects_wrong_length(client):
    assert client.post("/predict", json={"signal": [1.0, 2.0, 3.0]}).status_code == 422


def test_predict_rejects_nan(client):
    raw = '{"signal": [' + ",".join(["0.0"] * 2047 + ["NaN"]) + "]}"
    r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_rejects_oversized_body(client):
    r = client.post("/predict", content="x" * 300_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413