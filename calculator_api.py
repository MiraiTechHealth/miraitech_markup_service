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

import pandas as pd
from fastapi import Body, FastAPI, HTTPException


BACKEND_ROOT = Path(__file__).resolve().parent.parent / "MiraiTech-backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from jump_bilstm_runtime import MarkupJumpBiLSTMCalculator  # noqa: E402
from step_detector_ttest_runtime import StepDetectorTTest  # noqa: E402


app = FastAPI(title="MiraiTech Markup Calculators")

CALCULATOR_LABELS = {
    "step-detector-ttest": "Step Detector T-Test",
    "tkeo-cadence": "TKEO Cadence",
    "step-cadence": "Step Cadence",
    "jump-metrics": "Jump BiLSTM",
    "force-jump": "Bilateral GRF",
    "protocol-walking-detector": "Walking Test Detector",
    "protocol-running-detector": "Running Analysis Detector",
    "protocol-jumping-detector": "Jump Analysis Detector",
    "protocol-shuttle-detector": "Shuttle Run Detector",
    "protocol-sprint-detector": "Sprint 30 m Detector",
    "protocol-beep-detector": "Beep Test Detector",
    "protocol-ttest-detector": "T-Test Detector",
}

CALCULATOR_MODELS = {
    "step-detector-ttest": "Pressure peak detector",
    "tkeo-cadence": "TKEO + peak detection",
    "step-cadence": "StepResUNet",
    "jump-metrics": "JumpBiLSTM",
    "force-jump": "BiLSTMCNNRegressor",
    "protocol-walking-detector": "StepResUNet contacts",
    "protocol-running-detector": "StepResUNet contacts",
    "protocol-jumping-detector": "JumpBiLSTM flight detector",
    "protocol-shuttle-detector": "TurnCalculator shuttle phases",
    "protocol-sprint-detector": "CausalSpeedTCN + StepResUNet sprint steps",
    "protocol-beep-detector": "YoyoTurnCalculator phases",
    "protocol-ttest-detector": "TurnCalculator T-Test phases",
}

CALCULATOR_MODEL_FILES = {
    "step-cadence": "step_gc_model.pt",
    "jump-metrics": "jump_bilstm.pt",
    "force-jump": "fz_bilateral.pt",
    "protocol-walking-detector": "step_gc_model.pt",
    "protocol-running-detector": "step_gc_model.pt",
    "protocol-jumping-detector": "jump_bilstm.pt",
    "protocol-sprint-detector": "speed_cont_v5.pt + step_gc_model.pt",
}

PER_FOOT_TURN_DETECTOR_IDS = {
    "protocol-shuttle-detector",
    "protocol-beep-detector",
    "protocol-ttest-detector",
}

SENSOR_TO_FOOT = {
    "ESP32_Sensor_1": "left",
    "ESP32_Sensor_2": "right",
}

_markup_jump_bilstm_calculator = None


def _get_markup_jump_calculator():
    """Create and reuse the 500 Hz, 24-feature JumpBiLSTM calculator."""
    global _markup_jump_bilstm_calculator
    if _markup_jump_bilstm_calculator is None:
        _markup_jump_bilstm_calculator = MarkupJumpBiLSTMCalculator()
    return _markup_jump_bilstm_calculator


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
    """Run the JumpBiLSTM and expose compact UI-friendly metrics."""
    rows = _rows_with_time_in_ms(rows)
    calculator = _get_markup_jump_calculator()
    # Keep detected pairs and aggregate metrics from the same inference pass.
    # The JumpBiLSTM calculator owns its full 24-feature preprocessing pipeline.
    with calculator._infer_lock:
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
            flight_time_ms = float(pair["flight_time_ms"])
            jump_height_cm = pair.get("jump_height_cm")
            if jump_height_cm is None:
                jump_height_cm = calculator.calculate_jump_height_from_flight_time(
                    flight_time_ms
                )
            contacts.append({
                "foot": foot,
                "start_time_s": start_ms / 1000.0,
                "end_time_s": end_ms / 1000.0,
                "peak_time_s": start_ms / 1000.0,
                "duration_ms": flight_time_ms,
                "jump_height_cm": _round_or_none(jump_height_cm),
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


def _protocol_contact_detector_result(
    calculator_id: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Expose StepResUNet contact regions as walking/running detections."""
    base = _cadence_result("step-cadence", rows)
    contacts = list(base.get("contacts") or [])
    left_count = sum(1 for event in contacts if event.get("foot") == "left")
    right_count = sum(1 for event in contacts if event.get("foot") == "right")
    return {
        "calculator": calculator_id,
        "label": CALCULATOR_LABELS[calculator_id],
        "model": CALCULATOR_MODELS[calculator_id],
        "model_file": CALCULATOR_MODEL_FILES.get(calculator_id),
        "contacts": contacts,
        "summary": {
            "event_count": len(contacts),
            "contact_count": len(contacts),
            "left_count": left_count,
            "right_count": right_count,
            "is_valid": bool(base.get("summary", {}).get("is_valid")),
        },
    }


def _protocol_jump_detector_result(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Expose JumpBiLSTM take-off-to-landing regions without metric cards."""
    calculator_id = "protocol-jumping-detector"
    base = _jump_result(rows)
    contacts = list(base.get("contacts") or [])
    return {
        "calculator": calculator_id,
        "label": CALCULATOR_LABELS[calculator_id],
        "model": CALCULATOR_MODELS[calculator_id],
        "model_file": CALCULATOR_MODEL_FILES.get(calculator_id),
        "contacts": contacts,
        "summary": {
            "event_count": len(contacts),
            "flight_count": len(contacts),
            "left_count": sum(1 for event in contacts if event.get("foot") == "left"),
            "right_count": sum(1 for event in contacts if event.get("foot") == "right"),
            "is_valid": bool(base.get("summary", {}).get("is_valid")),
        },
    }


def _session_time_bounds_ms(rows: List[Dict[str, Any]]) -> tuple[float | None, float | None]:
    values = []
    for row in rows:
        try:
            values.append(float(row.get("Time")))
        except (TypeError, ValueError):
            continue
    return (min(values), max(values)) if values else (None, None)


def _turn_phase_contacts(
    turns: Iterable[Any],
    session_start_ms: float | None,
    session_end_ms: float | None,
    event_foot: str = "both",
) -> List[Dict[str, Any]]:
    """Build alternating running and turning intervals for graph overlays."""
    ordered = sorted(turns, key=lambda turn: float(turn.start_time_ms))
    contacts: List[Dict[str, Any]] = []
    cursor = session_start_ms

    for phase_index, turn in enumerate(ordered):
        start_ms = float(turn.start_time_ms)
        end_ms = float(turn.end_time_ms)
        if cursor is not None and start_ms > cursor:
            contacts.append({
                "foot": event_foot,
                "start_time_s": cursor / 1000.0,
                "end_time_s": start_ms / 1000.0,
                "peak_time_s": cursor / 1000.0,
                "duration_ms": start_ms - cursor,
                "kind": "run",
                "phase_index": phase_index,
                "confidence": None,
            })
        contacts.append({
            "foot": event_foot,
            "start_time_s": start_ms / 1000.0,
            "end_time_s": end_ms / 1000.0,
            "peak_time_s": (start_ms + end_ms) / 2000.0,
            "duration_ms": float(turn.duration_ms),
            "kind": "turn",
            "phase_index": phase_index,
            "direction": str(turn.direction),
            "angle_deg": float(turn.angle_deg),
            "pivot_foot": turn.pivot_foot,
            "confidence": None,
        })
        cursor = max(cursor, end_ms) if cursor is not None else end_ms

    if ordered and cursor is not None and session_end_ms is not None and session_end_ms > cursor:
        contacts.append({
            "foot": event_foot,
            "start_time_s": cursor / 1000.0,
            "end_time_s": session_end_ms / 1000.0,
            "peak_time_s": cursor / 1000.0,
            "duration_ms": session_end_ms - cursor,
            "kind": "run",
            "phase_index": len(ordered),
            "confidence": None,
        })
    return contacts


def _rows_for_detection_foot(
    rows: List[Dict[str, Any]],
    detection_foot: str,
    sensor_name: str | None,
) -> tuple[List[Dict[str, Any]], str | None]:
    """Select one insole IMU without baking UI sensor ordering into detection."""
    if detection_foot == "both":
        return rows, None

    available_names = list(dict.fromkeys(
        str(row.get("Name"))
        for row in rows
        if row.get("Name") not in (None, "")
    ))
    selected_name = sensor_name if sensor_name in available_names else None

    if selected_name is None:
        expected_name = {
            "left": "ESP32_Sensor_1",
            "right": "ESP32_Sensor_2",
        }[detection_foot]
        if expected_name in available_names:
            selected_name = expected_name

    has_known_sensor_names = any(
        name in {"ESP32_Sensor_1", "ESP32_Sensor_2"}
        for name in available_names
    )
    if selected_name is None and not has_known_sensor_names:
        fallback_index = 0 if detection_foot == "left" else 1
        if fallback_index < len(available_names):
            selected_name = available_names[fallback_index]

    if selected_name is None:
        raise ValueError(f"No sensor data available for {detection_foot} foot")

    selected_rows = [row for row in rows if str(row.get("Name")) == selected_name]
    if not selected_rows:
        raise ValueError(f"Sensor {selected_name} has no rows")
    return selected_rows, selected_name


def _protocol_turn_detector_result(
    calculator_id: str,
    rows: List[Dict[str, Any]],
    detection_foot: str = "both",
    sensor_name: str | None = None,
) -> Dict[str, Any]:
    """Run one detector per requested foot; ``both`` overlays L and R results."""
    from app.services.calculators.turn_calculator import TurnCalculator

    def create_calculator():
        if calculator_id == "protocol-beep-detector":
            from app.services.calculators.yoyo_turn_calculator import YoyoTurnCalculator

            return YoyoTurnCalculator()
        if calculator_id == "protocol-ttest-detector":
            from app.services.ttest_analysis import TTEST_TURN_PARAMS

            return TurnCalculator(**TTEST_TURN_PARAMS)
        return TurnCalculator(
            min_change_deg=50,
            max_duration_ms=1200,
            enter_max_ms=400,
            exit_slow_ms=10,
            fast_rate_deg_per_ms=0.14,
        )

    requested_feet = ["left", "right"] if detection_foot == "both" else [detection_foot]
    contacts: List[Dict[str, Any]] = []
    foot_summaries: Dict[str, Dict[str, Any]] = {}
    sensor_names: Dict[str, str] = {}

    for foot in requested_feet:
        requested_sensor = sensor_name if detection_foot != "both" else None
        try:
            foot_rows, selected_sensor_name = _rows_for_detection_foot(
                rows,
                foot,
                requested_sensor,
            )
        except ValueError:
            if detection_foot != "both":
                raise
            continue

        normalised_rows = _rows_with_time_in_ms(foot_rows)
        calculator = create_calculator()
        result = calculator.calculate(normalised_rows)
        start_ms, end_ms = _session_time_bounds_ms(normalised_rows)
        foot_contacts = _turn_phase_contacts(
            result.turns,
            start_ms,
            end_ms,
            event_foot=foot,
        )
        turn_count = sum(1 for event in foot_contacts if event.get("kind") == "turn")
        run_count = sum(1 for event in foot_contacts if event.get("kind") == "run")
        contacts.extend(foot_contacts)
        if selected_sensor_name is not None:
            sensor_names[foot] = selected_sensor_name
        foot_summaries[foot] = {
            "event_count": len(foot_contacts),
            "turn_count": turn_count,
            "run_count": run_count,
            "is_valid": bool(result.is_valid),
        }

    if not foot_summaries:
        raise ValueError("No left or right foot sensor data available")

    contacts.sort(key=lambda event: (
        float(event.get("start_time_s", 0)),
        0 if event.get("foot") == "left" else 1,
    ))
    turn_count = sum(summary["turn_count"] for summary in foot_summaries.values())
    run_count = sum(summary["run_count"] for summary in foot_summaries.values())
    selected_sensor_name = sensor_names.get(detection_foot) if detection_foot != "both" else None
    return {
        "calculator": calculator_id,
        "label": CALCULATOR_LABELS[calculator_id],
        "model": CALCULATOR_MODELS[calculator_id],
        "model_file": CALCULATOR_MODEL_FILES.get(calculator_id),
        "contacts": contacts,
        "summary": {
            "event_count": len(contacts),
            "turn_count": turn_count,
            "run_count": run_count,
            "left_turn_count": foot_summaries.get("left", {}).get("turn_count", 0),
            "right_turn_count": foot_summaries.get("right", {}).get("turn_count", 0),
            "left_run_count": foot_summaries.get("left", {}).get("run_count", 0),
            "right_run_count": foot_summaries.get("right", {}).get("run_count", 0),
            "is_valid": any(summary["is_valid"] for summary in foot_summaries.values()),
            "detection_foot": detection_foot,
            "sensor_name": selected_sensor_name,
            "sensor_names": sensor_names,
        },
    }


def _distance_at_time(points: List[Any], time_ms: float) -> float | None:
    """Linearly interpolate cumulative SpeedTCN distance at one event time."""
    if not points:
        return None
    if time_ms <= float(points[0].time):
        return float(points[0].distance)
    if time_ms >= float(points[-1].time):
        return float(points[-1].distance)
    for left, right in zip(points, points[1:]):
        left_time = float(left.time)
        right_time = float(right.time)
        if left_time <= time_ms <= right_time:
            if right_time <= left_time:
                return float(left.distance)
            fraction = (time_ms - left_time) / (right_time - left_time)
            return float(left.distance) + fraction * (
                float(right.distance) - float(left.distance)
            )
    return None


def _valid_length(value: float | None, maximum: float) -> float | None:
    if value is None or value < 0.05 or value > maximum:
        return None
    return round(float(value), 3)


def _protocol_sprint_detector_result(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect 30 m, sprint steps, step length and same-foot stride length."""
    from app.services.calculators.ml_speed_calculator import get_speed_calculator

    calculator_id = "protocol-sprint-detector"
    normalised_rows = _rows_with_time_in_ms(rows)
    result = get_speed_calculator().calculate(normalised_rows)
    points = list(result.series[0]) if result.series else []
    start = next(
        (point for point in points if float(point.speed) > 0 and float(point.distance) > 0),
        None,
    )
    finish = None
    if start is not None:
        target_distance = float(start.distance) + 30.0
        finish = next(
            (point for point in points if float(point.time) > float(start.time)
             and float(point.distance) >= target_distance),
            None,
        )

    contacts: List[Dict[str, Any]] = []
    step_contacts: List[Dict[str, Any]] = []
    step_lengths: List[float] = []
    stride_lengths: List[float] = []
    stride_lengths_by_foot = {"left": [], "right": []}
    cadence_spm = None
    if start is not None and finish is not None:
        start_time_s = float(start.time) / 1000.0
        finish_time_s = float(finish.time) / 1000.0
        start_distance = float(start.distance)
        contacts.append({
            "foot": "both",
            "start_time_s": start_time_s,
            "end_time_s": finish_time_s,
            "peak_time_s": start_time_s,
            "duration_ms": float(finish.time) - float(start.time),
            "kind": "sprint",
            "distance_m": 30.0,
            "confidence": None,
        })

        cadence_result = _cadence_result("step-cadence", normalised_rows)
        cadence_spm = cadence_result.get("summary", {}).get("cadence_spm")
        detected_steps = sorted(
            (
                dict(event)
                for event in cadence_result.get("contacts", [])
                if start_time_s <= float(event.get("start_time_s", -1)) <= finish_time_s
            ),
            key=lambda event: float(event["start_time_s"]),
        )

        previous_distance = None
        previous_foot_distance: Dict[str, float] = {}
        for step_index, event in enumerate(detected_steps, start=1):
            event_time_ms = float(event["start_time_s"]) * 1000.0
            absolute_distance = _distance_at_time(points, event_time_ms)
            if absolute_distance is None:
                continue

            step_length = _valid_length(
                absolute_distance - previous_distance if previous_distance is not None else None,
                maximum=3.0,
            )
            foot = str(event.get("foot"))
            stride_length = _valid_length(
                absolute_distance - previous_foot_distance[foot]
                if foot in previous_foot_distance else None,
                maximum=4.0,
            )
            previous_distance = absolute_distance
            previous_foot_distance[foot] = absolute_distance

            if step_length is not None:
                step_lengths.append(step_length)
            if stride_length is not None:
                stride_lengths.append(stride_length)
                if foot in stride_lengths_by_foot:
                    stride_lengths_by_foot[foot].append(stride_length)

            event["end_time_s"] = min(float(event["end_time_s"]), finish_time_s)
            event["duration_ms"] = max(
                0.0,
                (float(event["end_time_s"]) - float(event["start_time_s"])) * 1000.0,
            )
            event.update({
                "kind": "step",
                "step_index": step_index,
                "distance_m": round(absolute_distance - start_distance, 3),
                "step_length_m": step_length,
                "stride_length_m": stride_length,
            })
            step_contacts.append(event)

        contacts.extend(step_contacts)

    return {
        "calculator": calculator_id,
        "label": CALCULATOR_LABELS[calculator_id],
        "model": CALCULATOR_MODELS[calculator_id],
        "model_file": CALCULATOR_MODEL_FILES.get(calculator_id),
        "contacts": contacts,
        "summary": {
            "event_count": len(contacts),
            "sprint_count": 1 if start is not None and finish is not None else 0,
            "step_count": len(step_contacts),
            "left_count": sum(1 for event in step_contacts if event.get("foot") == "left"),
            "right_count": sum(1 for event in step_contacts if event.get("foot") == "right"),
            "step_length_m": round(float(median(step_lengths)), 3) if step_lengths else None,
            "stride_length_m": round(float(median(stride_lengths)), 3) if stride_lengths else None,
            "stride_length_left_m": (
                round(float(median(stride_lengths_by_foot["left"])), 3)
                if stride_lengths_by_foot["left"] else None
            ),
            "stride_length_right_m": (
                round(float(median(stride_lengths_by_foot["right"])), 3)
                if stride_lengths_by_foot["right"] else None
            ),
            "cadence_spm": cadence_spm,
            "is_valid": bool(start is not None and finish is not None),
        },
    }


def _calculate(
    calculator_id: str,
    rows: List[Dict[str, Any]],
    *,
    detection_foot: str = "both",
    sensor_name: str | None = None,
) -> Dict[str, Any]:
    if calculator_id == "step-detector-ttest":
        return _ttest_result(rows)
    if calculator_id == "jump-metrics":
        return _jump_result(rows)
    if calculator_id in {"protocol-walking-detector", "protocol-running-detector"}:
        return _protocol_contact_detector_result(calculator_id, rows)
    if calculator_id == "protocol-jumping-detector":
        return _protocol_jump_detector_result(rows)
    if calculator_id in PER_FOOT_TURN_DETECTOR_IDS:
        return _protocol_turn_detector_result(
            calculator_id,
            rows,
            detection_foot=detection_foot,
            sensor_name=sensor_name,
        )
    if calculator_id == "protocol-sprint-detector":
        return _protocol_sprint_detector_result(rows)
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

    detection_foot = "both"
    sensor_name = None
    if calculator_id in PER_FOOT_TURN_DETECTOR_IDS:
        detection_foot = str(payload.get("detection_foot") or "both").lower()
        if detection_foot not in {"both", "left", "right"}:
            raise HTTPException(
                status_code=422,
                detail="detection_foot must be one of: both, left, right",
            )
        raw_sensor_name = payload.get("sensor_name")
        sensor_name = str(raw_sensor_name) if raw_sensor_name not in (None, "") else None

    try:
        return await asyncio.to_thread(
            _calculate,
            calculator_id,
            rows,
            detection_foot=detection_foot,
            sensor_name=sensor_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
