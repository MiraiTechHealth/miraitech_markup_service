"""Local calculator companion for the MiraiTech markup service.

This process lives with the markup tool and imports the existing calculator
implementations read-only from the sibling MiraiTech backend checkout. It does
not add or change any backend API routes.
"""

from __future__ import annotations

import asyncio
import json
import os
from statistics import median
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Response


DEFAULT_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "MiraiTech-backend"
BACKEND_ROOT = Path(
    os.environ.get("MIRAITECH_BACKEND_ROOT", str(DEFAULT_BACKEND_ROOT))
).expanduser().resolve()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from jump_bilstm_runtime import MarkupJumpBiLSTMCalculator  # noqa: E402
from plate_flight_gt import PlateFlightError, plate_flight_result  # noqa: E402
from app.utils.auth import get_current_user  # noqa: E402
from loguru import logger  # noqa: E402


app = FastAPI(title="MiraiTech Markup Calculators")

CALCULATOR_LABELS = {
    "step-detector-ttest": "Step Detector T-Test",
    "tkeo-cadence": "TKEO Cadence",
    "step-cadence": "Step Cadence",
    "jump-metrics": "Jump BiLSTM",
    "jump-events": "Jump events · plates",
    "force-jump": "Bilateral GRF",
    "grf-split": "Total GRF · plates",
    "plate-flight": "Полёт по плитам · разметка v7",
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
    "jump-events": "NewJumpModelByAdil (plate-trained TCN)",
    "force-jump": "JumpForceBW regressor",
    "grf-split": "JumpGRFTotal (plate-trained total regressor, 20 Hz target)",
    "plate-flight": "plate_flight_v7 (force-plate ground truth, no ML)",
    "protocol-walking-detector": "GCTTCN contacts",
    "protocol-running-detector": "GCTTCN contacts",
    "protocol-jumping-detector": "JumpBiLSTM flight detector",
    "protocol-shuttle-detector": "TurnCalculator shuttle phases",
    "protocol-sprint-detector": "CausalSpeedTCN + GCTTCN sprint steps",
    "protocol-beep-detector": "YoyoTurnCalculator phases",
    "protocol-ttest-detector": "TurnCalculator T-Test phases",
}
# The per-foot GRF curve is returned to the browser for plotting, so it is thinned
# to this many points. A 20 s session at 500 Hz is 10k samples; the graph cannot
# resolve more than a few thousand anyway, and the peaks in `events` are read off
# the full-rate curve before thinning.
GRF_SPLIT_MAX_POINTS = 4000

CALCULATOR_MODEL_FILES = {
    "step-cadence": "step_gc_model.pt",
    "jump-metrics": "jump_bilstm.pt",
    "jump-events": "new_jump_model_byAdil.pt",
    "force-jump": "jump_force_total.pt",
    "grf-split": "jump_grf_total.pt",
    "protocol-walking-detector": "gct_best.pt",
    "protocol-running-detector": "gct_best.pt",
    "protocol-jumping-detector": "jump_bilstm.pt",
    "protocol-sprint-detector": "speed_cont_v5.pt + gct_best.pt",
}

# Movement one-hot the plate-trained jump model is conditioned on. It is an
# input channel, so the same session scored under a different protocol gives
# different events — the operator picks it, and "vert" is the default because it
# is both the most common markup case and the protocol the model is strongest on.
JUMP_EVENT_PROTOCOLS = ("vert", "fwd_sl", "side_sl", "sl_hop", "mv3", "mv5", "mv6")
DEFAULT_JUMP_EVENT_PROTOCOL = "vert"

PER_FOOT_TURN_DETECTOR_IDS = {
    "protocol-shuttle-detector",
    "protocol-beep-detector",
    "protocol-ttest-detector",
}

# Calculators that read the session column-wise and so never need the row dicts.
# On a 121k-row export ``to_dict`` alone cost ~530 ms — more than the model it
# was feeding — so the frame is handed over untouched.
FRAME_CALCULATOR_IDS = {
    "jump-events",
    "plate-flight",
}

SENSOR_TO_FOOT = {
    "ESP32_Sensor_1": "left",
    "ESP32_Sensor_2": "right",
}

# Every calculator keys its feet off the firmware sensor names. The force-plate
# research corpus labels the same rows "Right Foot" / "Left Foot", so a parquet
# loaded straight from it used to come out with zero feet and an empty prediction
# — no error in the UI, just contacts: []. Normalise the aliases here, at the API
# boundary, so it costs no rewrite of the corpus and covers every calculator.
NAME_ALIASES = {
    "right foot": "ESP32_Sensor_2",
    "right": "ESP32_Sensor_2",
    "r": "ESP32_Sensor_2",
    "left foot": "ESP32_Sensor_1",
    "left": "ESP32_Sensor_1",
    "l": "ESP32_Sensor_1",
}


def _canonical_sensor_name(name: Any) -> Any:
    """"Right Foot" -> "ESP32_Sensor_2"; anything already canonical is untouched."""
    if not isinstance(name, str):
        return name
    return NAME_ALIASES.get(name.strip().lower(), name)


def _canonicalise_names_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "Name" not in df.columns:
        return df
    mapped = df["Name"].map(_canonical_sensor_name)
    if mapped.equals(df["Name"]):
        return df
    renamed = sorted({str(a) for a, b in zip(df["Name"], mapped) if a != b})
    logger.info(f"markup: Name aliases mapped to sensor names: {renamed}")
    df = df.copy()
    df["Name"] = mapped
    return df


def _canonicalise_names_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out = []
    for row in rows:
        name = row.get("Name")
        canon = _canonical_sensor_name(name)
        if canon == name:
            out.append(row)
            continue
        seen.add(str(name))
        item = dict(row)
        item["Name"] = canon
        out.append(item)
    if seen:
        logger.info(f"markup: Name aliases mapped to sensor names: {sorted(seen)}")
    return out


_markup_jump_bilstm_calculator = None

GOOGLE_CLOUD_AUTH_HINT = (
    "Нет доступа к Google Cloud. Выполните `gcloud auth application-default login` "
    "в отдельном терминале, затем перезапустите `npm run dev`."
)


async def _run_markup_io(func, *args):
    """Run blocking markup storage I/O and turn expired ADC into a useful response."""
    try:
        return await asyncio.to_thread(func, *args)
    except HTTPException:
        raise
    except Exception as exc:
        message = str(exc).lower()
        auth_markers = (
            "reauthentication is needed",
            "default credentials were not found",
            "could not automatically determine credentials",
            "invalid_grant",
        )
        if any(marker in message for marker in auth_markers):
            raise HTTPException(status_code=503, detail=GOOGLE_CLOUD_AUTH_HINT) from exc
        raise HTTPException(
            status_code=500,
            detail="Локальный API разметчика не смог получить данные сессии.",
        ) from exc


def _parse_additional_info(raw: Any) -> Dict[str, Any]:
    """Return a session's additional_info as a plain object."""
    value = raw
    try:
        while isinstance(value, str):
            value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _list_markup_sessions(search: str | None, page_size: int) -> Dict[str, Any]:
    """List sessions across owners for the internal markup workspace."""
    from app.core.config import settings
    from app.db.database import get_db

    where = ""
    params: list[Any] = []
    if search:
        where = """
            WHERE CAST(s.session_id AS TEXT) LIKE %s
               OR LOWER(COALESCE(s.session_title, '')) LIKE %s
               OR LOWER(COALESCE(p.patient_name, '')) LIKE %s
        """
        pattern = f"%{search.strip().lower()}%"
        params.extend([pattern, pattern, pattern])
    params.append(page_size)

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT s.session_id, s.user_id, s.date, s.time, s.session_title,
                       s.device_id, s.protocol_id, s.time_offset,
                       p.patient_name, pd.protocol_name
                FROM {settings.DB_SCHEMA}.sessions s
                LEFT JOIN {settings.DB_SCHEMA}.members p
                       ON p.member_id = s.member_id
                LEFT JOIN {settings.DB_SCHEMA}.protocol_dict pd
                       ON pd.protocol_id = s.protocol_id
                {where}
                ORDER BY s.session_id DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()

    return {
        "items": [
            {
                "id": row["session_id"],
                "owner_id": row.get("user_id"),
                "member_name": row.get("patient_name") or "—",
                "session_title": row.get("session_title"),
                "date": row.get("date"),
                "time": row.get("time"),
                "device_id": row.get("device_id"),
                "protocol_id": row.get("protocol_id"),
                "protocol_name": row.get("protocol_name"),
                "time_offset": row.get("time_offset"),
            }
            for row in rows
        ]
    }


def _get_markup_session(session_id: int) -> Dict[str, Any]:
    """Load lightweight metadata without an owner/account restriction."""
    from app.core.config import settings
    from app.db.database import get_db

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT s.session_id, s.user_id, s.date, s.time, s.session_title,
                       s.additional_info, s.device_id, s.protocol_id,
                       s.time_offset, p.patient_name, pd.protocol_name
                FROM {settings.DB_SCHEMA}.sessions s
                LEFT JOIN {settings.DB_SCHEMA}.members p
                       ON p.member_id = s.member_id
                LEFT JOIN {settings.DB_SCHEMA}.protocol_dict pd
                       ON pd.protocol_id = s.protocol_id
                WHERE s.session_id = %s
                """,
                (session_id,),
            )
            row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": row["session_id"],
        "owner_id": row.get("user_id"),
        "member_name": row.get("patient_name") or "—",
        "session_title": row.get("session_title"),
        "date": row.get("date"),
        "time": row.get("time"),
        "device_id": row.get("device_id"),
        "protocol_id": row.get("protocol_id"),
        "protocol_name": row.get("protocol_name"),
        "time_offset": row.get("time_offset"),
        "additional_info": _parse_additional_info(row.get("additional_info")),
    }


def _get_markup_sprint_charts(session_id: int) -> Any:
    """Build sprint charts without restricting the session to the token owner."""
    from app.services.charts_service import ChartsService

    session = _get_markup_session(session_id)
    owner_id = session.get("owner_id") or 0
    return ChartsService().get_sprint_charts(session_id, owner_id)


def _update_markup_additional_info(
    session_id: int,
    additional_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Replace markup metadata without coupling it to the signed-in owner."""
    from app.core.config import settings
    from app.db.database import get_db
    from app.db.redis_sync import invalidate_session_caches

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {settings.DB_SCHEMA}.sessions
                SET additional_info = %s
                WHERE session_id = %s
                RETURNING additional_info
                """,
                (json.dumps(additional_info), session_id),
            )
            row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    invalidate_session_caches(session_id)
    return {
        "id": session_id,
        "additional_info": _parse_additional_info(row.get("additional_info")),
    }


def _update_markup_session_title(session_id: int, session_title: str) -> Dict[str, Any]:
    """Rename a session from the markup workspace, regardless of its owner."""
    from app.core.config import settings
    from app.db.database import get_db
    from app.db.redis_sync import invalidate_session_caches

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {settings.DB_SCHEMA}.sessions
                SET session_title = %s
                WHERE session_id = %s
                RETURNING session_title
                """,
                (session_title or None, session_id),
            )
            row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    invalidate_session_caches(session_id)
    return {"id": session_id, "session_title": row.get("session_title")}


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
    for row in rows[:300]:
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


def _frame_with_time_in_ms(df: pd.DataFrame) -> pd.DataFrame:
    """``_rows_with_time_in_ms`` for the column-wise path — same 0.5 ms rule."""
    if "Time" not in df.columns:
        return df
    head = pd.to_numeric(df["Time"].head(300), errors="coerce").dropna()
    ordered = np.unique(head.to_numpy(float))
    deltas = np.diff(ordered)
    deltas = deltas[deltas > 0]
    if deltas.size == 0 or float(np.median(deltas)) >= 0.5:
        return df
    out = df.copy()
    out["Time"] = pd.to_numeric(out["Time"], errors="coerce") * 1000.0
    return out


def _cadence_result(
    calculator_id: str,
    rows: List[Dict[str, Any]],
    calculator: Any = None,
) -> Dict[str, Any]:
    """Turn a cadence detector's contact regions into markup overlays.

    ``calculator`` lets a caller supply its own detector; walking, running and
    the sprint protocol pass the GCTTCN-backed adapter the backend uses for
    those protocols.
    """
    if calculator is None:
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


def _jump_events_result(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    protocol: str = DEFAULT_JUMP_EVENT_PROTOCOL,
) -> Dict[str, Any]:
    """Run the plate-trained jump detector and expose its bilateral events.

    Imported lazily: this model lives on a backend branch, and a checkout without
    it should cost this one calculator, not the whole companion API.

    Events here are **bilateral** — the model was trained against force plates,
    where flight starts when the last foot leaves the ground and ends when the
    first foot touches down. So each jump is one interval, not a left and a right
    one, and ``foot`` is left unset: the overlay is drawn once across the chart
    without a per-foot time shift.
    """
    from app.services.calculators.new_jump_model_byAdil_calculator import (
        get_new_jump_model_byAdil_calculator,
    )

    data = (_frame_with_time_in_ms(data) if isinstance(data, pd.DataFrame)
            else _rows_with_time_in_ms(data))
    calculator = get_new_jump_model_byAdil_calculator()
    result = calculator.calculate(data, protocol=protocol)

    contacts = [
        {
            "foot": None,
            "start_time_s": jump.takeoff_time_ms / 1000.0,
            "end_time_s": jump.landing_time_ms / 1000.0,
            "peak_time_s": jump.takeoff_time_ms / 1000.0,
            "duration_ms": jump.flight_time_ms,
            "jump_height_cm": _round_or_none(jump.jump_height_cm),
            "contact_time_ms": _round_or_none(jump.contact_time_ms, 1),
            "rsi": _round_or_none(jump.rsi, 3),
            "kind": "flight",
            "confidence": None,
        }
        for jump in result.jumps
    ]

    summary = result.summary
    return {
        "calculator": "jump-events",
        "label": CALCULATOR_LABELS["jump-events"],
        "model": CALCULATOR_MODELS["jump-events"],
        "model_file": CALCULATOR_MODEL_FILES["jump-events"],
        "contacts": contacts,
        "summary": {
            "protocol": result.protocol,
            "total_jump_count": summary.n_jumps,
            "event_count": summary.n_jumps,
            "flight_count": summary.n_jumps,
            "mean_flight_time_ms": _round_or_none(summary.mean_flight_time_ms, 1),
            "max_flight_time_ms": _round_or_none(summary.max_flight_time_ms, 1),
            "mean_jump_height_cm": _round_or_none(summary.mean_jump_height_cm),
            "max_jump_height_cm": _round_or_none(summary.max_jump_height_cm),
            "mean_contact_time_ms": _round_or_none(summary.mean_contact_time_ms, 1),
            "mean_rsi": _round_or_none(summary.mean_rsi, 3),
            "is_valid": bool(summary.is_valid),
        },
    }


def _plate_flight_result(df: pd.DataFrame) -> Dict[str, Any]:
    """Force-plate ground truth of the jump detector, for sessions that carry the
    plate force as a column (``Plate_Fz_N`` from ``export_markup_parquet.py`` or
    ``1:Fz`` / ``2:Fz`` from the synced research corpus).

    No model runs here: this is the labeler the ``jump-events`` TCN is trained on
    — both plates below 20 N, stitched over 10 ms blips, floored at 60 ms,
    rejected when an insole is loaded (athlete beside the plate), accepted on an
    IMU impact within 60 ms of landing OR both feet in free fall (v6), and with
    jumps across the plate edge (onto / off the plates, only one edge visible)
    masked as unknown rather than ground (v7). Contacts are bilateral like
    ``jump-events`` and use ``kind`` ``plate_flight`` for jumps and ``plate_mask``
    for segments the labeler refused to call either way; a mask's ``status`` and
    ``hop`` say why.
    """
    df = _frame_with_time_in_ms(df)
    return plate_flight_result(
        df, CALCULATOR_LABELS["plate-flight"], CALCULATOR_MODELS["plate-flight"])


def _force_units(units: Any) -> Dict[str, Any] | None:
    """Flatten a JumpForceUnits into the markup response's scalar fields."""
    if units is None:
        return None
    return {
        "percent_bw": _round_or_none(units.percent_bw, 1),
        "bw": _round_or_none(units.bw, 3),
        "n": _round_or_none(units.n, 1),
        "kg": _round_or_none(units.kg, 1),
        "lb": _round_or_none(units.lb, 1),
    }


def _force_result(rows: List[Dict[str, Any]], weight_kg: float) -> Dict[str, Any]:
    """Run bilateral Fz regression and expose peak/flight metrics.

    Forces come out of the model as % of body weight; the calculator derives N,
    kgf and lbf from the weight passed here, so the absolute units are only as
    good as that weight. Jump instants come from the jump detector, not from the
    force curve, so the events line up one-for-one with the jump metrics.
    """
    from app.services.calculators.jump_force_bw_calculator import (
        get_jump_force_bw_calculator,
    )

    rows = _rows_with_time_in_ms(rows)
    calculator = get_jump_force_bw_calculator()
    result = calculator.calculate(rows, weight_kg=weight_kg)
    peak = result.peak_force
    return {
        "calculator": "force-jump",
        "label": CALCULATOR_LABELS["force-jump"],
        "model": f"{CALCULATOR_MODELS['force-jump']} · {calculator.arch}",
        "model_file": CALCULATOR_MODEL_FILES["force-jump"],
        "events": [
            {
                "jump_index": event.jump_index,
                "takeoff_time_ms": _round_or_none(event.takeoff_time_ms),
                "landing_time_ms": _round_or_none(event.landing_time_ms),
                "flight_time_ms": _round_or_none(event.flight_time_ms),
                "feet": list(event.feet),
                "takeoff_force": _force_units(event.takeoff_force),
                "landing_force": _force_units(event.landing_force),
            }
            for event in result.events
        ],
        "summary": {
            "peak_force_n": _round_or_none(peak.n, 1) if peak else None,
            "peak_force_bw": _round_or_none(peak.bw, 3) if peak else None,
            "peak_force_percent_bw": _round_or_none(peak.percent_bw, 1) if peak else None,
            "peak_takeoff_force": _force_units(result.peak_takeoff_force),
            "peak_landing_force": _force_units(result.peak_landing_force),
            "avg_takeoff_force": _force_units(result.avg_takeoff_force),
            "avg_landing_force": _force_units(result.avg_landing_force),
            "jump_count": result.n_jumps,
            "weight_kg": _round_or_none(result.weight_kg),
            "weight_source": result.weight_source,
            "is_valid": bool(result.is_valid),
        },
    }


def _grf_split_result(rows: List[Dict[str, Any]], weight_kg: float,
                      protocol: str | None = None) -> Dict[str, Any]:
    """Total Fz curve plus per-jump forces, for eyeballing the model on a session.

    The model's target is the plate **low-passed at ``target_lowpass_hz``** (20 Hz);
    ``export_markup_parquet.py`` writes ``Plate_Fz_total_lp20_pctBW`` so the two can
    be laid over each other in the same definition. Against the raw plate column the
    landing peak reads ~10% low by design.

    Unlike ``_force_result`` this returns the **curves** as ``data_points``, so the
    markup graph can draw the prediction over the pressure traces it was computed
    from — which is the whole point of having the model here rather than only in
    the API. The series is thinned to keep the payload sane; peaks are read off
    the full-rate curve inside the calculator, before thinning, so a thinned
    sample never becomes a reported peak.

    Take-off and landing come from the plate-trained bilateral detector, the same
    one the ``jump-events`` calculator exposes, so the flight bands drawn by both
    line up.
    """
    from app.services.calculators.jump_grf_split_calculator import (
        get_jump_grf_split_calculator,
    )

    rows = _rows_with_time_in_ms(rows)
    calculator = get_jump_grf_split_calculator()
    result = calculator.calculate(rows, weight_kg=weight_kg, protocol=protocol)

    step = max(1, len(result.times_ms) // GRF_SPLIT_MAX_POINTS)
    data_points = [
        {"time": result.times_ms[i], "total": result.fz_total_percent_bw[i]}
        for i in range(0, len(result.times_ms), step)
    ]

    summary = result.summary
    test_card = (result.model_metrics or {}).get("test") or {}
    return {
        "calculator": "grf-split",
        "label": CALCULATOR_LABELS["grf-split"],
        "model": f"{CALCULATOR_MODELS['grf-split']} · {result.arch}",
        "model_file": CALCULATOR_MODEL_FILES["grf-split"],
        "data_points": data_points,
        "sample_step": step,
        "contacts": [
            {
                "foot": None,
                "start_ms": jump.takeoff_time_ms,
                "end_ms": jump.landing_time_ms,
                "duration_ms": jump.flight_time_ms,
                "kind": "flight",
                "confidence": None,
            }
            for jump in result.jumps
        ],
        "events": [
            {
                "jump_index": jump.jump_index,
                "takeoff_time_ms": jump.takeoff_time_ms,
                "landing_time_ms": jump.landing_time_ms,
                "flight_time_ms": jump.flight_time_ms,
                "pushoff_force": _force_units(jump.pushoff_force),
                "landing_force": _force_units(jump.landing_force),
                "contact_impulse_bw_s": jump.contact_impulse_bw_s,
                "contact_time_ms": jump.contact_time_ms,
            }
            for jump in result.jumps
        ],
        "summary": {
            "jump_count": summary.n_jumps,
            "peak_force": _force_units(summary.peak_force),
            "peak_pushoff_force": _force_units(summary.peak_pushoff_force),
            "peak_landing_force": _force_units(summary.peak_landing_force),
            "avg_pushoff_force": _force_units(summary.avg_pushoff_force),
            "avg_landing_force": _force_units(summary.avg_landing_force),
            "mean_contact_impulse_bw_s": summary.mean_contact_impulse_bw_s,
            # The definition the curve is in: compare with a plate filtered the same way.
            "target_lowpass_hz": result.target_lowpass_hz,
            "weight_kg": _round_or_none(summary.weight_kg),
            "weight_source": summary.weight_source,
            "event_source": result.event_source,
            "n_samples": result.n_samples,
            "is_valid": bool(summary.is_valid),
            # What the plates said about these weights, on athletes they never saw.
            # Shown beside the numbers so nobody reads a landing peak as measured.
            "held_out_contact_rmse_pctbw": _round_or_none(
                test_card.get("rmse_contact_pctbw"), 1),
            "held_out_landing_peak_mae_pctbw": _round_or_none(
                test_card.get("peak_landing_mae_pctbw"), 1),
            "held_out_landing_peak_bias_pctbw": _round_or_none(
                test_card.get("peak_landing_bias_pctbw"), 1),
            "held_out_landing_peak_raw_bias_pctbw": _round_or_none(
                test_card.get("peak_landing_raw_bias_pctbw"), 1),
        },
    }


def _protocol_contact_detector_result(
    calculator_id: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Expose GCTTCN contact regions as walking/running detections."""
    from app.services.calculators.step_cadence_calculator import get_gct_cadence_calculator

    base = _cadence_result("step-cadence", rows, get_gct_cadence_calculator())
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
        for row in rows[:500]
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
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    detection_foot: str = "both",
    sensor_name: str | None = None,
    protocol: str = DEFAULT_JUMP_EVENT_PROTOCOL,
) -> Dict[str, Any]:
    """Run one detector per requested foot; ``both`` overlays L and R results."""
    from app.schemas.turn_cod import TurnEvent
    from app.services.calculators.turn_calculator import TurnCalculator

    def create_calculator():
        if calculator_id == "protocol-beep-detector":
            return TurnCalculator(
                min_change_deg=100,
                max_duration_ms=2000,
            )
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

    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if "Time" in df.columns or "time" in df.columns:
        tcol = "Time" if "Time" in df.columns else "time"
        s = pd.to_numeric(df[tcol].iloc[:300], errors="coerce").dropna()
        if not s.empty:
            diffs = np.diff(np.sort(s.values))
            diffs = diffs[diffs > 0]
            if len(diffs) and np.median(diffs) < 0.5:
                df = df.copy()
                df[tcol] = pd.to_numeric(df[tcol], errors="coerce") * 1000.0

    requested_feet = ["left", "right"] if detection_foot == "both" else [detection_foot]
    contacts: List[Dict[str, Any]] = []
    foot_summaries: Dict[str, Dict[str, Any]] = {}
    sensor_names: Dict[str, str] = {}

    available_names = [str(n) for n in df["Name"].dropna().unique()] if "Name" in df.columns else []

    for foot in requested_feet:
        selected_sensor_name = sensor_name if detection_foot != "both" else None
        if selected_sensor_name is None:
            expected_name = "ESP32_Sensor_1" if foot == "left" else "ESP32_Sensor_2"
            if expected_name in available_names:
                selected_sensor_name = expected_name
            elif len(available_names) > 0:
                idx = 0 if foot == "left" else min(1, len(available_names) - 1)
                selected_sensor_name = available_names[idx]

        if selected_sensor_name is not None and "Name" in df.columns:
            foot_df = df[df["Name"] == selected_sensor_name]
        else:
            foot_df = df

        if foot_df.empty:
            if detection_foot != "both":
                raise ValueError(f"No sensor data available for {foot} foot")
            continue

        calculator = create_calculator()
        raw_turns = calculator.identify(foot_df, group_sensors=False)

        turn_events: List[Any] = []
        for t in raw_turns:
            duration = t["end_time"] - t["start_time"]
            if duration <= 0:
                continue
            angle = t["angle"]
            direction = "left" if angle < 0 else "right" if angle > 0 else "unknown"
            mean_vel = (angle / (duration / 1000.0)) if duration > 0 else 0.0
            turn_events.append(TurnEvent(
                index=t["index"],
                start_time_ms=t["start_time"],
                end_time_ms=t["end_time"],
                duration_ms=duration,
                angle_deg=angle,
                direction=direction,
                mean_angular_velocity_deg_s=round(mean_vel, 2),
                pivot_foot=foot,
                turning_foot="right" if foot == "left" else "left",
                sensor_name=selected_sensor_name,
            ))

        if calculator.merge_close_turns_ms is not None and calculator.merge_close_turns_ms > 0:
            turn_events = calculator._merge_close_turns(turn_events)
        if calculator.max_turns is not None and len(turn_events) > calculator.max_turns:
            turn_events = calculator._trim_to_max_turns(turn_events)

        tcol = "Time" if "Time" in foot_df.columns else ("time" if "time" in foot_df.columns else None)
        if tcol and not foot_df[tcol].empty:
            s_t = pd.to_numeric(foot_df[tcol], errors="coerce").dropna()
            start_ms = float(s_t.min()) if not s_t.empty else None
            end_ms = float(s_t.max()) if not s_t.empty else None
        else:
            start_ms = end_ms = None

        foot_contacts = _turn_phase_contacts(
            turn_events,
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
            "is_valid": len(turn_events) > 0,
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
    from app.services.calculators.step_cadence_calculator import get_gct_cadence_calculator

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

    end = finish
    if start is not None and end is None:
        end = next(
            (
                point for point in reversed(points)
                if float(point.time) > float(start.time)
                and float(point.distance) >= float(start.distance)
            ),
            None,
        )

    contacts: List[Dict[str, Any]] = []
    step_contacts: List[Dict[str, Any]] = []
    step_lengths: List[float] = []
    stride_lengths: List[float] = []
    stride_lengths_by_foot = {"left": [], "right": []}
    cadence_spm = None
    segment_distance_m = None
    if start is not None and end is not None:
        start_time_s = float(start.time) / 1000.0
        end_time_s = float(end.time) / 1000.0
        start_distance = float(start.distance)
        is_complete = finish is not None
        segment_distance_m = (
            30.0
            if is_complete
            else max(0.0, float(end.distance) - start_distance)
        )
        contacts.append({
            "foot": "both",
            "start_time_s": start_time_s,
            "end_time_s": end_time_s,
            "peak_time_s": start_time_s,
            "duration_ms": float(end.time) - float(start.time),
            "kind": "sprint",
            "distance_m": round(segment_distance_m, 3),
            "is_complete": is_complete,
            "confidence": None,
        })

        cadence_result = _cadence_result(
            "step-cadence", normalised_rows, get_gct_cadence_calculator()
        )
        cadence_spm = cadence_result.get("summary", {}).get("cadence_spm")
        # Every step the cadence model found across the whole recording is kept
        # for markup - this is a QA tool, not the timing product, so a step
        # before the start gate or after the 30 m finish must stay visible and
        # keep its real GCT. Only the 30 m-dash summary stats (step_count,
        # median step/stride length, cadence) are scoped to the timed segment.
        detected_steps = sorted(
            (dict(event) for event in cadence_result.get("contacts", [])),
            key=lambda event: float(event["start_time_s"]),
        )

        previous_distance = None
        previous_foot_distance: Dict[str, float] = {}
        segment_step_index = 0
        for event in detected_steps:
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

            in_segment = start_time_s <= float(event["start_time_s"]) <= end_time_s
            if in_segment:
                segment_step_index += 1
                if step_length is not None:
                    step_lengths.append(step_length)
                if stride_length is not None:
                    stride_lengths.append(stride_length)
                    if foot in stride_lengths_by_foot:
                        stride_lengths_by_foot[foot].append(stride_length)

            event.update({
                "kind": "step",
                "step_index": segment_step_index if in_segment else None,
                "in_segment": in_segment,
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
            "segment_found": bool(start is not None and end is not None),
            "distance_m": round(segment_distance_m, 3) if segment_distance_m is not None else None,
            "step_count": sum(1 for event in step_contacts if event.get("in_segment")),
            "left_count": sum(1 for event in step_contacts if event.get("foot") == "left" and event.get("in_segment")),
            "right_count": sum(1 for event in step_contacts if event.get("foot") == "right" and event.get("in_segment")),
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
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    *,
    detection_foot: str = "both",
    sensor_name: str | None = None,
    protocol: str = DEFAULT_JUMP_EVENT_PROTOCOL,
) -> Dict[str, Any]:
    if calculator_id in PER_FOOT_TURN_DETECTOR_IDS:
        return _protocol_turn_detector_result(
            calculator_id,
            data,
            detection_foot=detection_foot,
            sensor_name=sensor_name,
        )

    if calculator_id == "jump-events":
        return _jump_events_result(data, protocol)

    rows = data if isinstance(data, list) else data.to_dict(orient="records")
    if calculator_id == "step-detector-ttest":
        return _ttest_result(rows)
    if calculator_id == "jump-metrics":
        return _jump_result(rows)
    if calculator_id in {"protocol-walking-detector", "protocol-running-detector"}:
        return _protocol_contact_detector_result(calculator_id, rows)
    if calculator_id == "protocol-jumping-detector":
        return _protocol_jump_detector_result(rows)
    if calculator_id == "protocol-sprint-detector":
        return _protocol_sprint_detector_result(rows)
    return _cadence_result(calculator_id, rows)


IMU_COLUMNS = ["AcX", "AcY", "AcZ", "XData", "YData", "ZData"]

GRAVITY_MS2 = 9.80665


class InsoleAHRS:
    """Madgwick AHRS that turns raw insole IMU samples into firmware channels.

    Raw sensors report accelerations with gravity still in them and gyroscope
    rates in dps. The new firmware (miraitech_sensors.h) instead ships linear
    accelerations in the BLE basis (AcX = linY, AcY = -linX, AcZ = linZ) plus
    an integrated Heading, Roll and Pitch, each quantised to its own step.
    """

    def __init__(
        self,
        sample_rate: float = 500.0,
        beta: float = 0.1,
        initial_yaw: float = 0.0,
    ):
        self.dt = 1.0 / sample_rate
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.integrated_yaw = float(initial_yaw)
        self.initialized = False

    def init_orientation(self, ax: float, ay: float, az: float) -> None:
        norm_a = np.sqrt(ax * ax + ay * ay + az * az)
        if norm_a > 1e-4:
            ax_n, ay_n, az_n = ax / norm_a, ay / norm_a, az / norm_a
            roll = np.arctan2(ay_n, az_n)
            pitch = np.arctan2(-ax_n, np.sqrt(ay_n * ay_n + az_n * az_n))
            cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
            cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
            self.q = np.array([cr * cp, sr * cp, cr * sp, -sr * sp])
            self.q /= np.linalg.norm(self.q)
        self.initialized = True

    def update(
        self,
        ax: float,
        ay: float,
        az: float,
        gx_dps: float,
        gy_dps: float,
        gz_dps: float,
    ):
        if not self.initialized:
            self.init_orientation(ax, ay, az)

        gx, gy, gz = np.deg2rad(gx_dps), np.deg2rad(gy_dps), np.deg2rad(gz_dps)
        q0, q1, q2, q3 = self.q

        # Quaternion rate from the gyroscope
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        norm_a = np.sqrt(ax * ax + ay * ay + az * az)
        if norm_a > 1e-4:
            ax_n, ay_n, az_n = ax / norm_a, ay / norm_a, az / norm_a
            _2q0, _2q1, _2q2, _2q3 = 2.0 * q0, 2.0 * q1, 2.0 * q2, 2.0 * q3
            _4q0, _4q1, _4q2 = 4.0 * q0, 4.0 * q1, 4.0 * q2
            _8q1, _8q2 = 8.0 * q1, 8.0 * q2
            q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3

            s0 = _4q0 * q2q2 + _2q2 * ax_n + _4q0 * q1q1 - _2q1 * ay_n
            s1 = (
                _4q1 * q3q3
                - _2q3 * ax_n
                + 4.0 * q0q0 * q1
                - _2q0 * ay_n
                - _4q1
                + _8q1 * q1q1
                + _8q1 * q2q2
                + _4q1 * az_n
            )
            s2 = (
                4.0 * q0q0 * q2
                + _2q0 * ax_n
                + _4q2 * q3q3
                - _2q3 * ay_n
                - _4q2
                + _8q2 * q1q1
                + _8q2 * q2q2
                + _4q2 * az_n
            )
            s3 = 4.0 * q1q1 * q3 - _2q1 * ax_n + 4.0 * q2q2 * q3 - _2q2 * ay_n

            norm_s = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            if norm_s > 1e-4:
                qDot1 -= self.beta * (s0 / norm_s)
                qDot2 -= self.beta * (s1 / norm_s)
                qDot3 -= self.beta * (s2 / norm_s)
                qDot4 -= self.beta * (s3 / norm_s)

        q0 += qDot1 * self.dt
        q1 += qDot2 * self.dt
        q2 += qDot3 * self.dt
        q3 += qDot4 * self.dt
        norm_q = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        self.q = np.array([q0, q1, q2, q3]) / norm_q
        q0, q1, q2, q3 = self.q

        # Gravity vector in the body frame
        gx_body = 2.0 * (q1 * q3 - q0 * q2)
        gy_body = 2.0 * (q0 * q1 + q2 * q3)
        gz_body = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

        # Linear body accelerations in m/s²
        lin_ax = ax - gx_body * GRAVITY_MS2
        lin_ay = ay - gy_body * GRAVITY_MS2
        lin_az = az - gz_body * GRAVITY_MS2

        # Euler angles
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = np.rad2deg(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (q0 * q2 - q3 * q1)
        pitch = (
            np.sign(sinp) * 90.0
            if np.abs(sinp) >= 1.0
            else np.rad2deg(np.arcsin(sinp))
        )

        # Heading integrated from the gyroscope projected on gravity
        yawRateDps = gx_dps * gx_body + gy_dps * gy_body + gz_dps * gz_body
        self.integrated_yaw = (self.integrated_yaw - yawRateDps * self.dt) % 360.0

        # Pack into the miraitech_sensors.h layout and quantisation steps
        AcX = np.round(lin_ay * 25.0) / 25.0
        AcY = np.round(-lin_ax * 25.0) / 25.0
        AcZ = np.round(lin_az * 25.0) / 25.0

        hdg = (int(np.round(self.integrated_yaw)) % 360) - 180
        XData = float(hdg + 180)
        YData = float(int(np.round(roll * 0.5)) * 2)
        ZData = float(int(np.round(-pitch)))

        return AcX, AcY, AcZ, XData, YData, ZData


def is_raw_sensor(sensor_df: pd.DataFrame) -> bool:
    """Report whether a sensor still carries raw IMU signals.

    Raw accelerations keep gravity (|AcZ| stays near 9.8) and raw gyroscope
    rates go negative and well past the ±90° an angle channel can hold.
    """
    if len(sensor_df) == 0:
        return False
    if not {"AcZ", "XData", "ZData"}.issubset(sensor_df.columns):
        return False
    acz = pd.to_numeric(sensor_df["AcZ"], errors="coerce")
    xdata = pd.to_numeric(sensor_df["XData"], errors="coerce")
    zdata = pd.to_numeric(sensor_df["ZData"], errors="coerce")
    has_gravity = bool(acz.abs().median() > 5.0)
    has_raw_gyro = bool((xdata.min() < -5.0) or (zdata.abs().max() > 95.0))
    return has_gravity or has_raw_gyro


def _preprocess_imu_dataframe(
    df: pd.DataFrame,
    target_sensor: str = "auto",
    sample_rate: float = 500.0,
) -> tuple[pd.DataFrame, List[str]]:
    df_out = df.copy()
    processed_sensors: List[str] = []

    # Seed the heading from a sensor that already runs the new firmware, so
    # both feet share one course reference.
    initial_yaw = 0.0
    for _name, group in df_out.groupby("Name"):
        if "XData" in group.columns and not is_raw_sensor(group):
            seed = pd.to_numeric(group["XData"], errors="coerce").dropna()
            if not seed.empty:
                initial_yaw = float(seed.iloc[0])
                break

    for name, group in df_out.groupby("Name"):
        name_str = str(name)
        if target_sensor == "auto":
            needs_processing = is_raw_sensor(group)
        elif target_sensor == "all":
            needs_processing = True
        else:
            needs_processing = name_str == target_sensor
        if not needs_processing:
            continue

        ordered = group.sort_values("Time") if "Time" in group.columns else group
        channels = {}
        for column in IMU_COLUMNS:
            if column in ordered.columns:
                values = pd.to_numeric(ordered[column], errors="coerce")
                channels[column] = values.fillna(0.0).to_numpy(dtype=float)
            else:
                channels[column] = np.zeros(len(ordered))

        ahrs = InsoleAHRS(
            sample_rate=sample_rate,
            beta=0.1,
            initial_yaw=initial_yaw,
        )
        processed_rows = [
            ahrs.update(
                ax=channels["AcX"][index],
                ay=channels["AcY"][index],
                az=channels["AcZ"][index],
                gx_dps=channels["XData"][index],
                gy_dps=channels["YData"][index],
                gz_dps=channels["ZData"][index],
            )
            for index in range(len(ordered))
        ]
        if not processed_rows:
            continue

        # Assign by the sorted index: rows of a sensor are not necessarily
        # stored in time order inside the session frame.
        df_out.loc[ordered.index, IMU_COLUMNS] = np.array(processed_rows, dtype=float)
        processed_sensors.append(name_str)

    return df_out, processed_sensors


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/markup/sessions")
async def list_markup_sessions(
    search: str | None = Query(None),
    page_size: int = Query(100, ge=1, le=500),
    _current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return sessions from every account for the internal markup tool."""
    return await _run_markup_io(_list_markup_sessions, search, page_size)


@app.get("/markup/sessions/{session_id}")
async def get_markup_session(
    session_id: int,
    _current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return account-independent session metadata used by the markup UI."""
    return await _run_markup_io(_get_markup_session, session_id)


@app.get("/markup/sessions/{session_id}/parquet")
async def get_markup_session_parquet(
    session_id: int,
    _current_user: dict = Depends(get_current_user),
) -> Response:
    """Serve the GCS Parquet object (with the backend's legacy DB fallback)."""
    from app.services.session_data_storage import get_session_parquet_bytes

    data = await _run_markup_io(get_session_parquet_bytes, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session has no data")
    return Response(
        content=data,
        media_type="application/vnd.apache.parquet",
        headers={
            "Content-Disposition": f'attachment; filename="session_{session_id}.parquet"',
            "X-Markup-Storage": "gcs-first",
        },
    )


@app.get("/markup/sessions/{session_id}/charts/sprint")
async def get_markup_session_sprint_charts(
    session_id: int,
    _current_user: dict = Depends(get_current_user),
) -> Any:
    """Return sprint predictions for any session visible to the markup workspace."""
    return await _run_markup_io(_get_markup_sprint_charts, session_id)


@app.put("/markup/sessions/{session_id}/additional-info")
async def update_markup_additional_info(
    session_id: int,
    payload: Dict[str, Any] = Body(...),
    _current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Save shared markup metadata independently of the session owner."""
    additional_info = payload.get("additional_info")
    if not isinstance(additional_info, dict):
        raise HTTPException(status_code=422, detail="additional_info must be an object")
    return await _run_markup_io(
        _update_markup_additional_info,
        session_id,
        additional_info,
    )


@app.put("/markup/sessions/{session_id}/title")
async def update_markup_session_title(
    session_id: int,
    payload: Dict[str, Any] = Body(...),
    _current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Edit the session title shown in the markup header."""
    raw_title = payload.get("session_title")
    if raw_title is not None and not isinstance(raw_title, str):
        raise HTTPException(status_code=422, detail="session_title must be a string")
    session_title = (raw_title or "").strip()
    if len(session_title) > 255:
        raise HTTPException(status_code=422, detail="session_title is too long (max 255)")
    return await _run_markup_io(
        _update_markup_session_title,
        session_id,
        session_title,
    )


def _extract_session_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    columns = payload.get("columns")
    if isinstance(columns, dict) and columns:
        return _canonicalise_names_frame(pd.DataFrame(columns))
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        return _canonicalise_names_frame(pd.DataFrame(rows))
    raise HTTPException(status_code=422, detail="Session columns or rows are required")


def _extract_session_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = payload.get("columns")
    if isinstance(columns, dict) and columns:
        return _canonicalise_names_rows(pd.DataFrame(columns).to_dict(orient="records"))
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        return _canonicalise_names_rows(rows)
    raise HTTPException(status_code=422, detail="Session columns or rows are required")


@app.post("/markup/preprocess-imu")
async def preprocess_imu(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Convert raw IMU channels of a session into the new firmware format."""
    df = _extract_session_dataframe(payload)

    target_sensor = str(payload.get("target_sensor") or "auto")
    raw_sample_rate = payload.get("sample_rate")
    try:
        sample_rate = 500.0 if raw_sample_rate is None else float(raw_sample_rate)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="sample_rate must be a number")
    if sample_rate <= 0:
        raise HTTPException(status_code=422, detail="sample_rate must be positive")

    if "Name" not in df.columns:
        raise HTTPException(status_code=422, detail="Session data has no Name column")
    if target_sensor not in {"auto", "all"}:
        known = {str(name) for name in df["Name"].dropna().unique()}
        if target_sensor not in known:
            raise HTTPException(
                status_code=422,
                detail=f"Датчик {target_sensor} отсутствует в данных сессии",
            )

    try:
        df_processed, processed_sensors = await asyncio.to_thread(
            _preprocess_imu_dataframe,
            df,
            target_sensor,
            sample_rate,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # NaN is not valid JSON — hand missing cells back as null.
    df_processed = df_processed.astype(object).where(pd.notna(df_processed), None)
    return {
        "columns": {col: df_processed[col].tolist() for col in df_processed.columns},
        "rows": df_processed.to_dict(orient="records"),
        "processed_sensors": processed_sensors,
        "success": True,
    }


@app.post("/calculate/{calculator_id}")
async def calculate(
    calculator_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    if calculator_id not in CALCULATOR_LABELS:
        raise HTTPException(status_code=404, detail="Unknown calculator")

    if calculator_id in PER_FOOT_TURN_DETECTOR_IDS or calculator_id in FRAME_CALCULATOR_IDS:
        data = _extract_session_dataframe(payload)
    else:
        data = _extract_session_rows(payload)

    if calculator_id == "force-jump":
        try:
            weight_kg = float(payload.get("weight_kg"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="weight_kg is required for Bilateral GRF")
        if weight_kg <= 0:
            raise HTTPException(status_code=422, detail="weight_kg must be positive")
        try:
            return await asyncio.to_thread(_force_result, data, weight_kg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if calculator_id == "grf-split":
        try:
            weight_kg = float(payload.get("weight_kg"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422,
                                detail="weight_kg is required for Total GRF")
        if weight_kg <= 0:
            raise HTTPException(status_code=422, detail="weight_kg must be positive")
        raw_protocol = payload.get("protocol")
        grf_protocol = (str(raw_protocol) if raw_protocol not in (None, "")
                        else DEFAULT_JUMP_EVENT_PROTOCOL)
        if grf_protocol not in JUMP_EVENT_PROTOCOLS:
            raise HTTPException(
                status_code=422,
                detail=f"protocol must be one of: {', '.join(JUMP_EVENT_PROTOCOLS)}")
        try:
            return await asyncio.to_thread(_grf_split_result, data, weight_kg, grf_protocol)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if calculator_id == "plate-flight":
        try:
            return await asyncio.to_thread(_plate_flight_result, data)
        except PlateFlightError as exc:
            # Not a failure of the tool: the session has no plates, or one foot.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    protocol = DEFAULT_JUMP_EVENT_PROTOCOL
    if calculator_id == "jump-events":
        raw_protocol = payload.get("protocol")
        protocol = str(raw_protocol) if raw_protocol not in (None, "") else DEFAULT_JUMP_EVENT_PROTOCOL
        if protocol not in JUMP_EVENT_PROTOCOLS:
            raise HTTPException(
                status_code=422,
                detail=f"protocol must be one of: {', '.join(JUMP_EVENT_PROTOCOLS)}",
            )

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
            data,
            detection_foot=detection_foot,
            sensor_name=sensor_name,
            protocol=protocol,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
