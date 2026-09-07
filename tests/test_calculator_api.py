"""Smoke tests for the markup companion API against a real force-plate session.

Run from the backend checkout, which owns the Python environment the API runs in:

    cd ../MiraiTech-backend && poetry run pytest ../miraitech_markup_service/tests -q

or ``npm run test:api`` from this directory.

The session is the synced research recording ``sportsmen_sync/synced_7896.parquet``
(12 vertical jumps, two force plates). Everything is skipped when it is not on
disk, and the calculators whose model bundle is missing from the backend skip
individually - the API is meant to degrade one calculator at a time.
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

MARKUP_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(
    os.environ.get("MIRAITECH_BACKEND_ROOT", str(MARKUP_ROOT.parent / "MiraiTech-backend"))
).resolve()
SESSION_PARQUET = Path(
    os.environ.get(
        "MARKUP_TEST_PARQUET",
        str(MARKUP_ROOT.parent / "jump_model" / "sportsmen_sync" / "synced_7896.parquet"),
    )
)
PARQUET = {"Content-Type": "application/vnd.apache.parquet"}
EXPECTED_JUMPS = 12

pytestmark = pytest.mark.skipif(
    not SESSION_PARQUET.is_file(), reason=f"test session missing: {SESSION_PARQUET}"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def api() -> httpx.Client:
    """The companion API as ``npm run dev`` starts it, with its warm-up finished."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "calculator_api:app",
         "--app-dir", str(MARKUP_ROOT), "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(BACKEND_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=240)
    deadline = time.monotonic() + 240
    try:
        while True:
            try:
                health = client.get("/health").json()
                if health.get("models_ready"):
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("calculator API did not come up")
            time.sleep(0.25)
        client.unavailable = set(health.get("unavailable", []))  # type: ignore[attr-defined]
        yield client
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def session_bytes() -> bytes:
    return SESSION_PARQUET.read_bytes()


def _needs(api: httpx.Client, calculator_id: str) -> None:
    if calculator_id in api.unavailable:  # type: ignore[attr-defined]
        pytest.skip(f"{calculator_id}: model bundle missing in the backend checkout")


def _post(api: httpx.Client, calculator_id: str, body: bytes, **params) -> httpx.Response:
    return api.post(f"/calculate/{calculator_id}", params=params, content=body, headers=PARQUET)


def _bands(result: dict) -> list:
    return [(round(c["start_time_s"], 3), round(c["end_time_s"], 3)) for c in result["contacts"]]


CONTACT_KEYS = {"foot", "start_time_s", "end_time_s", "peak_time_s", "duration_ms", "kind", "confidence"}


def test_health_reports_warm_up(api):
    health = api.get("/health").json()
    assert health["ok"] is True
    assert health["models_ready"] is True


def test_unknown_and_removed_calculators_are_404(api, session_bytes):
    for calculator_id in ("tkeo-cadence", "jump-metrics", "force-jump", "protocol-sprint-detector", "nope"):
        assert _post(api, calculator_id, session_bytes).status_code == 404, calculator_id


def test_jump_events_bytes_transport(api, session_bytes):
    _needs(api, "jump-events")
    result = _post(api, "jump-events", session_bytes, protocol="vert")
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["summary"]["total_jump_count"] == EXPECTED_JUMPS
    assert data["summary"]["protocol"] == "vert"
    assert all(CONTACT_KEYS <= set(c) for c in data["contacts"])
    assert all(c["foot"] is None and c["kind"] == "flight" for c in data["contacts"])


def test_json_transport_matches_bytes(api, session_bytes):
    _needs(api, "jump-events")
    frame = pd.read_parquet(io.BytesIO(session_bytes))
    columns = {c: frame[c].tolist() for c in (
        "Name", "Time", "AcX", "AcY", "AcZ", "XData", "YData", "ZData",
        "Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4")}
    via_json = api.post("/calculate/jump-events", json={"columns": columns, "protocol": "vert"})
    via_bytes = _post(api, "jump-events", session_bytes, protocol="vert")
    assert via_json.status_code == 200, via_json.text
    assert via_json.json()["contacts"] == via_bytes.json()["contacts"]


def test_result_cache_answers_repeats(api, session_bytes):
    _needs(api, "jump-events")
    first = _post(api, "jump-events", session_bytes, protocol="vert").json()
    started = time.perf_counter()
    second = _post(api, "jump-events", session_bytes, protocol="vert")
    assert second.status_code == 200
    assert second.json() == first
    assert time.perf_counter() - started < 0.5, "a repeat should come from the result cache"


def test_grf_shares_the_jump_detector_bands(api, session_bytes):
    _needs(api, "grf-split")
    _needs(api, "jump-events")
    jumps = _post(api, "jump-events", session_bytes, protocol="vert").json()
    grf = _post(api, "grf-split", session_bytes, protocol="vert", weight_kg=75)
    assert grf.status_code == 200, grf.text
    data = grf.json()
    assert data["summary"]["jump_count"] == EXPECTED_JUMPS
    assert data["summary"]["protocol"] == "vert"
    assert data["summary"]["event_source"] == "new_jump_model_byAdil"
    assert _bands(data) == _bands(jumps), "GRF flight bands must be the detector's"
    assert all(CONTACT_KEYS <= set(c) for c in data["contacts"])
    assert data["data_points"] and {"time", "total"} <= set(data["data_points"][0])
    assert data["summary"]["weight_kg"] == 75.0


def test_grf_reuses_caller_jump_pairs(api, session_bytes):
    _needs(api, "grf-split")
    _needs(api, "jump-events")
    jumps = _post(api, "jump-events", session_bytes, protocol="vert").json()
    pairs = json.dumps([[c["start_time_s"], c["end_time_s"]] for c in jumps["contacts"]])
    with_pairs = _post(api, "grf-split", session_bytes, protocol="vert", weight_kg=75, jump_pairs=pairs).json()
    detected = _post(api, "grf-split", session_bytes, protocol="vert", weight_kg=75).json()
    assert with_pairs["summary"]["event_source"] == "caller"
    assert _bands(with_pairs) == _bands(detected)
    assert with_pairs["summary"]["peak_landing_force"] == detected["summary"]["peak_landing_force"]


def test_grf_validation(api, session_bytes):
    _needs(api, "grf-split")
    assert _post(api, "grf-split", session_bytes, protocol="vert").status_code == 422
    assert _post(api, "grf-split", session_bytes, protocol="vert", weight_kg=-1).status_code == 422
    assert _post(api, "grf-split", session_bytes, weight_kg=70, jump_pairs="nope").status_code == 422
    assert _post(api, "jump-events", session_bytes, protocol="xx").status_code == 422


def test_step_detector_ttest(api, session_bytes):
    result = _post(api, "step-detector-ttest", session_bytes)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["contacts"], "the T-test step detector finds contacts on a jump session too"
    assert {c["foot"] for c in data["contacts"]} <= {"left", "right"}
    assert all(CONTACT_KEYS <= set(c) for c in data["contacts"])


def test_walking_detector(api, session_bytes):
    _needs(api, "protocol-walking-detector")
    result = _post(api, "protocol-walking-detector", session_bytes)
    assert result.status_code == 200, result.text
    summary = result.json()["summary"]
    assert summary["contact_count"] == summary["left_count"] + summary["right_count"] > 0


def test_turn_detectors_per_foot(api, session_bytes):
    both = _post(api, "protocol-ttest-detector", session_bytes, detection_foot="both")
    assert both.status_code == 200, both.text
    assert both.json()["summary"]["detection_foot"] == "both"
    left = _post(api, "protocol-shuttle-detector", session_bytes, detection_foot="left")
    assert left.status_code == 200, left.text
    assert left.json()["summary"]["detection_foot"] == "left"
    assert _post(api, "protocol-beep-detector", session_bytes, detection_foot="up").status_code == 422


def test_plate_flight_ground_truth(api, session_bytes):
    result = _post(api, "plate-flight", session_bytes)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["summary"]["total_jump_count"] > 0
    assert {c["kind"] for c in data["contacts"]} <= {"plate_flight", "plate_mask"}


def test_bad_bodies_are_422(api):
    assert api.post("/calculate/jump-events", content=b"not parquet", headers=PARQUET).status_code == 422
    assert api.post("/calculate/jump-events", content=b"", headers=PARQUET).status_code == 422
    assert api.post("/calculate/jump-events", json={}).status_code == 422
    assert api.post("/calculate/jump-events", json=[1, 2]).status_code == 422
