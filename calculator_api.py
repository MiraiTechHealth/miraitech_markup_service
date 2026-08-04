"""Local calculator companion for the MiraiTech markup service.

This process lives with the markup tool and imports the existing calculator
implementations read-only from the sibling MiraiTech backend checkout. It does
not add or change any backend API routes.
"""

from __future__ import annotations

import asyncio
from statistics import median
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException


BACKEND_ROOT = Path(__file__).resolve().parent.parent / "MiraiTech-backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


app = FastAPI(title="MiraiTech Markup Calculators")

CALCULATOR_LABELS = {
    "step-detector-ttest": "Step Detector T-Test",
    "tkeo-cadence": "TKEO Cadence",
    "step-cadence": "Step Cadence",
    "jump-metrics": "Jump CNN-LSTM",
    "force-jump": "Bilateral GRF",
}

CALCULATOR_MODELS = {
    "step-detector-ttest": "Pressure peak detector",
    "tkeo-cadence": "TKEO + peak detection",
    "step-cadence": "StepResUNet",
    "jump-metrics": "JumpCNNLSTM",
    "force-jump": "BiLSTMCNNRegressor",
}

CALCULATOR_MODEL_FILES = {
    "step-cadence": "step_gc_model.pt",
    "jump-metrics": "jump_cnn_lstm.pt",
    "force-jump": "fz_bilateral.pt",
}

SENSOR_TO_FOOT = {
    "ESP32_Sensor_1": "left",
    "ESP32_Sensor_2": "right",
}

_JUMP_RAW_COLUMNS = ("AcX", "AcY", "AcZ", "XData", "YData", "ZData", "GravityZ")
_JUMP_FEATURE_COLUMNS = ("AcZ", "AcX", "AcY", "XData", "YData", "ZData", "GravityZ")


def _preprocess_jump_foot(
    foot_data: List[Dict[str, Any]],
    foot_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare one foot's raw rows exactly as the JumpCNNLSTM expects.

    The model's backend calculator historically performed this transformation
    internally. The markup calculator API now performs it explicitly and passes
    the result through an adapter, while the base calculator still owns the
    trained scaler and sliding-window inference.
    """
    rows = []
    for entry in foot_data:
        try:
            time = float(entry.get("Time", 0))
            values = [float(entry.get(column, 0)) for column in _JUMP_RAW_COLUMNS]
            rows.append([time] + values)
        except (TypeError, ValueError):
            continue

    if not rows:
        return np.array([], dtype=np.float64), np.empty((0, len(_JUMP_RAW_COLUMNS)), dtype=np.float32)

    rows.sort(key=lambda row: row[0])
    data = np.asarray(rows, dtype=np.float64)

    # Columns after Time: AcX AcY AcZ XData YData ZData GravityZ.
    acx_index, acy_index = 1, 2
    position_indices = (4, 5, 6)

    if foot_type == "right":
        data[:, acx_index] = -data[:, acx_index]
        data[:, acy_index] = -data[:, acy_index]

    for column_index in position_indices:
        data[:, column_index] -= data[0, column_index]

    features_by_name = {
        "AcX": data[:, 1],
        "AcY": data[:, 2],
        "AcZ": data[:, 3],
        "XData": data[:, 4],
        "YData": data[:, 5],
        "ZData": data[:, 6],
        "GravityZ": data[:, 7],
    }
    features = np.column_stack(
        [features_by_name[column] for column in _JUMP_FEATURE_COLUMNS]
    ).astype(np.float32)
    return data[:, 0], features


_markup_jump_calculator = None


def _get_markup_jump_calculator():
    """Create one JumpCNNLSTM instance whose raw parser is markup-owned."""
    global _markup_jump_calculator
    if _markup_jump_calculator is None:
        from app.services.calculators.ml_jump_metrics_calculator import (
            MLJumpMetricsCalculator,
        )

        class MarkupPreprocessedJumpMetricsCalculator(MLJumpMetricsCalculator):
            def __init__(self, *args, **kwargs):
                self._markup_preprocessed_features = {}
                super().__init__(*args, **kwargs)

            def set_preprocessed_features(self, features_by_foot):
                self._markup_preprocessed_features = features_by_foot

            def _preprocess_foot(self, foot_data, foot_type):
                return self._markup_preprocessed_features.get(
                    foot_type,
                    (np.array([], dtype=np.float64), np.empty((0, 7), dtype=np.float32)),
                )

        _markup_jump_calculator = MarkupPreprocessedJumpMetricsCalculator(
            max_contact_time_ms=10000
        )
    return _markup_jump_calculator


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _foot_summary(stats: Any, events: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    events = list(events)
    confidences = [
        float(event["confidence"])
        for event in events
        if event.get("confidence") is not None
    ]
    return {
        "contact_count": int(stats.n_zero_runs),
        "mean_step_interval_s": (
            float(stats.mean_step_interval) if stats.mean_step_interval > 0 else None
        ),
        "mean_contact_duration_s": (
            float(stats.mean_contact_duration_s)
            if stats.mean_contact_duration_s > 0
            else None
        ),
        "mean_confidence": _mean_or_none(confidences),
    }


def _round_or_none(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _rows_with_time_in_ms(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise second-based markup files to the backend models' ms contract."""
    times = []
    for row in rows:
        try:
            times.append(float(row.get("Time")))
        except (TypeError, ValueError):
            continue

    ordered = sorted(set(times))
    deltas = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
    if not deltas or median(deltas) >= 0.5:
        return rows

    normalised = []
    for row in rows:
        item = dict(row)
        try:
            item["Time"] = float(item["Time"]) * 1000.0
        except (KeyError, TypeError, ValueError):
            pass
        normalised.append(item)
    return normalised


def _cadence_result(calculator_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if calculator_id == "tkeo-cadence":
        from app.services.calculators.tkeo_cadence_calculator import TkeoCadenceCalculator

        calculator = TkeoCadenceCalculator()
    else:
        from app.services.calculators.step_cadence_calculator import StepCadenceCalculator

        calculator = StepCadenceCalculator()

    result = calculator.calculate(rows)
    events_by_sensor = {
        sensor: calculator._viz_data.get(sensor, {}).get("contact_events", [])
        for sensor in SENSOR_TO_FOOT
    }
    contacts = []
    for sensor, foot in SENSOR_TO_FOOT.items():
        for event in events_by_sensor[sensor]:
            start_s = float(event["timestep_s"])
            duration_s = float(event["contact_time_s"])
            contacts.append(
                {
                    "foot": foot,
                    "start_time_s": start_s,
                    "end_time_s": start_s + duration_s,
                    "peak_time_s": start_s,
                    "duration_ms": duration_s * 1000.0,
                    "kind": "contact",
                    "confidence": (
                        float(event["confidence"])
                        if event.get("confidence") is not None
                        else None
                    ),
                }
            )

    contacts.sort(key=lambda contact: contact["start_time_s"])
    return {
        "calculator": calculator_id,
        "label": CALCULATOR_LABELS[calculator_id],
        "model": CALCULATOR_MODELS[calculator_id],
        "model_file": CALCULATOR_MODEL_FILES.get(calculator_id),
        "contacts": contacts,
        "summary": {
            "cadence_spm": float(result.cadence),
            "symmetry_index": float(result.symmetry_index),
            "gait_pattern": result.gait_pattern,
            "is_valid": bool(result.is_valid),
            "left": _foot_summary(result.left, events_by_sensor["ESP32_Sensor_1"]),
            "right": _foot_summary(result.right, events_by_sensor["ESP32_Sensor_2"]),
        },
    }


def _ttest_result(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from app.services.calculators.step_detector_ttest import StepDetectorTTest

    steps = StepDetectorTTest().calculate(pd.DataFrame(rows))
    contacts = [
        {
            "foot": str(row.foot),
            "start_time_s": float(row.t_start),
            "end_time_s": float(row.t_end),
            "peak_time_s": float(row.t_peak),
            "duration_ms": float(row.contact_ms),
            "kind": str(row.kind),
            "confidence": float(row.peak_z),
        }
        for row in steps.itertuples(index=False)
    ]

    def summary_for(foot: str) -> Dict[str, Any]:
        foot_rows = steps[steps["foot"] == foot]
        stride_s = [float(value) / 1000.0 for value in foot_rows["stride_ms"].dropna()]
        duration_s = [float(value) / 1000.0 for value in foot_rows["contact_ms"].dropna()]
        return {
            "contact_count": len(foot_rows),
            "mean_step_interval_s": _mean_or_none(stride_s),
            "mean_contact_duration_s": _mean_or_none(duration_s),
        }

    return {
        "calculator": "step-detector-ttest",
        "label": CALCULATOR_LABELS["step-detector-ttest"],
        "model": CALCULATOR_MODELS["step-detector-ttest"],
        "model_file": CALCULATOR_MODEL_FILES.get("step-detector-ttest"),
        "contacts": contacts,
        "summary": {
            "cadence_spm": None,
            "symmetry_index": None,
            "gait_pattern": None,
            "is_valid": None,
            "left": summary_for("left"),
            "right": summary_for("right"),
        },
    }


def _jump_result(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the JumpCNNLSTM and expose compact UI-friendly metrics."""
    rows = _rows_with_time_in_ms(rows)
    calculator = _get_markup_jump_calculator()
    preprocessed = {
        foot: _preprocess_jump_foot(
            [row for row in rows if row.get("Name") == sensor],
            foot,
        )
        for sensor, foot in SENSOR_TO_FOOT.items()
    }
    # Keep the detected pairs and the aggregate result from the same inference
    # pass. The calculator itself also uses this lock around calculate(). The
    # base calculator's scaler and sliding-window inference remain unchanged;
    # only its raw-row parser is bypassed to avoid preprocessing twice.
    with calculator._infer_lock:
        calculator.set_preprocessed_features(preprocessed)
        result = calculator.calculate(rows)
        details = {
            foot: list(calculator.analysis_details.get(foot, {}).get("jump_pairs", []))
            for foot in ("ESP32_Sensor_1", "ESP32_Sensor_2")
        }

    contacts = []
    for sensor, foot in (("ESP32_Sensor_1", "left"), ("ESP32_Sensor_2", "right")):
        for pair in details[sensor]:
            start_ms = float(pair["takeoff_time"])
            end_ms = float(pair["landing_time"])
            contacts.append({
                "foot": foot,
                "start_time_s": start_ms / 1000.0,
                "end_time_s": end_ms / 1000.0,
                "peak_time_s": start_ms / 1000.0,
                "duration_ms": float(pair["flight_time_ms"]),
                "kind": "flight",
                "confidence": None,
            })
    contacts.sort(key=lambda item: item["start_time_s"])

    flight = result.flight_time
    contact = result.contact_time
    rsi = result.rsi
    return {
        "calculator": "jump-metrics",
        "label": CALCULATOR_LABELS["jump-metrics"],
        "model": CALCULATOR_MODELS["jump-metrics"],
        "model_file": CALCULATOR_MODEL_FILES["jump-metrics"],
        "contacts": contacts,
        "summary": {
            "left_jump_count": len(flight.left_flight_times_ms),
            "right_jump_count": len(flight.right_flight_times_ms),
            "total_jump_count": int(flight.total_flight_events),
            "left_mean_flight_time_ms": _round_or_none(flight.left_mean_flight_time_ms),
            "right_mean_flight_time_ms": _round_or_none(flight.right_mean_flight_time_ms),
            "mean_jump_height_cm": _round_or_none(result.mean_jump_height_cm),
            "max_jump_height_cm": _round_or_none(result.max_jump_height_cm),
            "left_mean_contact_time_ms": _round_or_none(contact.left_mean_contact_time_ms),
            "right_mean_contact_time_ms": _round_or_none(contact.right_mean_contact_time_ms),
            "left_mean_rsi": _round_or_none(rsi.left_mean_rsi, 3),
            "right_mean_rsi": _round_or_none(rsi.right_mean_rsi, 3),
            "activity_type": result.activity_type,
            "is_valid": bool(result.is_valid),
        },
    }


def _force_result(rows: List[Dict[str, Any]], weight_kg: float) -> Dict[str, Any]:
    """Run bilateral Fz regression and expose peak/flight metrics."""
    from app.services.calculators.force_jump_calculator import get_force_jump_calculator

    rows = _rows_with_time_in_ms(rows)
    calculator = get_force_jump_calculator()
    result = calculator.calculate(rows, weight_kg=weight_kg)
    return {
        "calculator": "force-jump",
        "label": CALCULATOR_LABELS["force-jump"],
        "model": CALCULATOR_MODELS["force-jump"],
        "model_file": CALCULATOR_MODEL_FILES["force-jump"],
        "events": [
            {
                "takeoff_time_ms": _round_or_none(event.takeoff_time_ms),
                "landing_time_ms": _round_or_none(event.landing_time_ms),
                "flight_time_ms": _round_or_none(event.flight_time_ms),
            }
            for event in result.jump_events
        ],
        "summary": {
            "peak_force_n": _round_or_none(result.peak_force_n),
            "peak_force_bw": _round_or_none(result.peak_force_bw, 3),
            "jump_count": len(result.jump_events),
            "weight_kg": _round_or_none(result.weight_kg),
            "is_valid": bool(result.is_valid),
        },
    }


def _calculate(calculator_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if calculator_id == "step-detector-ttest":
        return _ttest_result(rows)
    if calculator_id == "jump-metrics":
        return _jump_result(rows)
    return _cadence_result(calculator_id, rows)


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/calculate/{calculator_id}")
async def calculate(
    calculator_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    if calculator_id not in CALCULATOR_LABELS:
        raise HTTPException(status_code=404, detail="Unknown calculator")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=422, detail="Session rows are required")

    if calculator_id == "force-jump":
        try:
            weight_kg = float(payload.get("weight_kg"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="weight_kg is required for Bilateral GRF")
        if weight_kg <= 0:
            raise HTTPException(status_code=422, detail="weight_kg must be positive")
        try:
            return await asyncio.to_thread(_force_result, rows, weight_kg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        return await asyncio.to_thread(_calculate, calculator_id, rows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
