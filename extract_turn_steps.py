#!/usr/bin/env python3
"""
Extract 3 steps before turn, 1 step after turn, and calculate GCT, Step Time, and Force for session data.

Self-contained script that runs without depending on specific backend repository branch versions.

Usage:
    python extract_turn_steps.py --session 6546 [--weight 70.0] [--output-csv turn_steps_session_6546.csv]
    python extract_turn_steps.py --file session_6546.parquet [--weight 70.0]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Set up paths to import MiraiTech backend if available
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_BACKEND_PATHS = [
    Path(os.environ.get("MIRAITECH_BACKEND_ROOT", "")),
    SCRIPT_DIR.parent / "MiraiTech-backend",
    SCRIPT_DIR.parent / "miraitech-backend",
    Path("/home/shared_folder/MiraiTech-backend"),
    Path("/home/shared_folder/miraitech-backend"),
    Path("/home/miraitech/MiraiTech-backend"),
    Path("/home/miraitech/miraitech-backend"),
    Path("/home/miraitech/app"),
]

# Auto-load .env file from backend directory so GCS / Settings can be read
try:
    from dotenv import load_dotenv
    for p in CANDIDATE_BACKEND_PATHS:
        if p and (p / ".env").exists():
            load_dotenv(p / ".env")
            break
except ImportError:
    pass

for p in CANDIDATE_BACKEND_PATHS:
    if p and p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Standalone Turn Detection (Yo-Yo / Beep Test detector)
# ─────────────────────────────────────────────────────────────────────────────
LEFT_FOOT = "ESP32_Sensor_1"
RIGHT_FOOT = "ESP32_Sensor_2"

class StandaloneTurnCalculator:
    """Self-contained Net-Rotation + Rise-Fraction turn detector for 180° shuttle turns."""

    def __init__(
        self,
        min_net_deg: float = 35.0,
        win_ms: float = 450.0,
        grid_ms: float = 2.0,
        pad_ms: float = 400.0,
        q_rise: float = 0.08,
        min_amp_deg: float = 100.0,  # 100° threshold for 180° Yo-Yo / Beep turns
        merge_gap_ms: float = 400.0,
        sg_window_length: int = 277,
        sg_polyorder: int = 2,
    ):
        self.min_net_deg = min_net_deg
        self.win_ms = win_ms
        self.grid_ms = grid_ms
        self.pad_ms = pad_ms
        self.q_rise = q_rise
        self.min_amp_deg = min_amp_deg
        self.merge_gap_ms = merge_gap_ms
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder

    @staticmethod
    def unwrap_angle_degrees(values: np.ndarray, threshold: float = 180.0) -> np.ndarray:
        if len(values) == 0:
            return values.astype(np.float64, copy=True)
        result = values.astype(np.float64, copy=True)
        offset = 0.0
        for i in range(1, len(values)):
            curr = values[i]
            prev = values[i - 1]
            if np.isnan(curr) or np.isnan(prev):
                continue
            diff = curr - prev
            if diff > threshold:
                offset -= 360.0
            elif diff < -threshold:
                offset += 360.0
            result[i] = float(values[i]) + offset
        return result

    def _savgol_smooth(self, th: np.ndarray) -> np.ndarray:
        n = len(th)
        poly = self.sg_polyorder
        wl = self.sg_window_length
        if wl > n:
            wl = n
        if wl % 2 == 0:
            wl -= 1
        if wl <= poly:
            wl = poly + 1
            if wl % 2 == 0:
                wl += 1
        if wl < 3 or wl > n or wl <= poly:
            return th
        return savgol_filter(th, window_length=wl, polyorder=poly)

    def _prepare_grid(self, time_arr: np.ndarray, angle_arr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        t = pd.to_numeric(pd.Series(time_arr), errors="coerce").to_numpy(dtype=float)
        a = pd.to_numeric(pd.Series(angle_arr), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(a)
        t, a = t[valid], a[valid]
        if len(t) < 10:
            return None

        order = np.argsort(t, kind="stable")
        t, a = t[order], a[order]
        theta = self.unwrap_angle_degrees(a)

        t_u, idx = np.unique(t, return_index=True)
        theta_u = theta[idx]
        if len(t_u) < 10 or t_u[-1] <= t_u[0]:
            return None

        unit = 1.0 if np.max(t_u) > 3600 else 1e-3
        grid_step = self.grid_ms * unit
        if grid_step <= 0:
            return None

        tg = np.arange(t_u[0], t_u[-1], grid_step)
        if len(tg) < 10:
            return None
        th = np.interp(tg, t_u, theta_u)
        th = self._savgol_smooth(th)
        return tg, th, unit

    def _locate_regions(self, tg: np.ndarray, th: np.ndarray, unit: float) -> List[Tuple[int, int]]:
        k = max(int(self.win_ms / self.grid_ms), 1)
        if k >= len(th):
            return []

        net = np.abs(th[k:] - th[:-k])
        mask = np.zeros(len(tg), dtype=bool)
        for i in np.flatnonzero(net >= self.min_net_deg):
            mask[i:i + k + 1] = True

        m = np.concatenate([[0], mask.astype(int), [0]])
        edges = np.flatnonzero(np.diff(m))
        regions = list(zip(edges[::2], edges[1::2] - 1))

        merge_gap = self.merge_gap_ms * unit
        merged: List[Tuple[int, int]] = []
        for s, e in regions:
            if merged and (tg[s] - tg[merged[-1][1]]) < merge_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged

    def _rise_boundaries(self, tg: np.ndarray, th: np.ndarray, s: int, e: int) -> Optional[Tuple[float, float, float]]:
        pad = int(self.pad_ms / self.grid_ms)
        q = self.q_rise
        lo_seg = th[max(s - pad, 0):s]
        hi_seg = th[e + 1:e + 1 + pad]
        if len(lo_seg) < 10 or len(hi_seg) < 10:
            return None

        lo, hi = float(np.median(lo_seg)), float(np.median(hi_seg))
        amp = hi - lo
        if abs(amp) < self.min_amp_deg:
            return None

        base = max(s - pad, 0)
        x = (th[base:e + 1 + pad] - lo) / amp
        mid = np.flatnonzero(x >= 0.5)
        if not len(mid):
            return None

        i = j = int(mid[0])
        while i > 0 and x[i] > q:
            i -= 1
        while j < len(x) - 1 and x[j] < 1 - q:
            j += 1
        return float(tg[base + i]), float(tg[base + j]), float(amp)

    def detect_turn_regions(self, time_arr: np.ndarray, angle_arr: np.ndarray) -> List[Tuple[float, float, float]]:
        prepared = self._prepare_grid(time_arr, angle_arr)
        if prepared is None:
            return []
        tg, th, unit = prepared
        out = []
        for s, e in self._locate_regions(tg, th, unit):
            bounds = self._rise_boundaries(tg, th, s, e)
            if bounds is not None:
                out.append(bounds)
        return out

    def identify(self, df: pd.DataFrame, group_sensors: bool = True) -> List[Dict[str, Any]]:
        tcol = "Time" if "Time" in df.columns else ("time" if "time" in df.columns else None)
        turn_col = "XData" if "XData" in df.columns else None
        if not tcol or not turn_col:
            return []

        if group_sensors and "Name" in df.columns:
            groups = [(name, group.sort_values(tcol).reset_index(drop=True)) for name, group in df.groupby("Name")]
        else:
            groups = [(None, df.sort_values(tcol).reset_index(drop=True))]

        per_sensor = []
        for name, group in groups:
            time_arr = group[tcol].to_numpy()
            angle_arr = group[turn_col].to_numpy()
            per_sensor.append({
                "name": name,
                "regions": self.detect_turn_regions(time_arr, angle_arr),
                "grid": self._prepare_grid(time_arr, angle_arr),
            })

        tagged = [
            {"start": s, "end": e, "angle": a, "grid": entry["grid"]}
            for entry in per_sensor
            for s, e, a in entry["regions"]
            if e > s
        ]
        if not tagged:
            return []
        tagged.sort(key=lambda t: t["start"])

        groups_tagged = [[tagged[0]]]
        group_end = tagged[0]["end"]
        for t in tagged[1:]:
            if t["start"] <= group_end:
                groups_tagged[-1].append(t)
                group_end = max(group_end, t["end"])
            else:
                groups_tagged.append([t])
                group_end = t["end"]

        results = []
        for i, grp in enumerate(groups_tagged):
            start = min(t["start"] for t in grp)
            end = max(t["end"] for t in grp)
            angle = grp[0]["angle"] if len(grp) == 1 else max(grp, key=lambda t: abs(t["angle"]))["angle"]
            results.append({
                "index": i,
                "start_time": int(round(start)),
                "end_time": int(round(end)),
                "angle": int(round(angle)),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Session Data Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_session_df(session_id: Optional[int], file_path: Optional[str]) -> pd.DataFrame:
    """Load session dataframe from file, backend service or direct GCS."""
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix in (".csv", ".txt"):
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    if session_id is not None:
        errors = []
        # 1. Try backend storage function if available
        try:
            from app.services.session_data_storage import load_session_dataframe
            df = load_session_dataframe(session_id)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            errors.append(f"load_session_dataframe: {exc}")

        # 2. Try load_session_data
        try:
            from app.services.session_data_storage import load_session_data
            rows = load_session_data(session_id)
            if rows:
                return pd.DataFrame(rows)
        except Exception as exc:
            errors.append(f"load_session_data: {exc}")

        # 3. Direct GCS download using google.cloud.storage across all potential prefixes
        try:
            from google.cloud import storage
            bucket_name = os.environ.get("GCS_BUCKET_NAME", "miraitech-sessions")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            candidate_prefixes = [
                os.environ.get("GCS_SESSION_PREFIX"),
                "dev_sessions",
                "sessions",
                "prod_sessions",
                "",
            ]
            for pr in candidate_prefixes:
                if pr is None:
                    continue
                blob_name = f"{pr}/{session_id}.parquet" if pr else f"{session_id}.parquet"
                try:
                    blob = bucket.blob(blob_name)
                    if blob.exists():
                        data_bytes = blob.download_as_bytes()
                        return pd.read_parquet(io.BytesIO(data_bytes))
                except Exception as b_exc:
                    errors.append(f"GCS {blob_name}: {b_exc}")
        except Exception as exc:
            errors.append(f"Direct GCS client: {exc}")

        # 4. Direct DB query fallback (if session data is in PostgreSQL column)
        try:
            from app.db.database import get_db
            from app.core.config import settings
            with get_db() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"SELECT data FROM {settings.DB_SCHEMA}.sessions WHERE session_id = %s",
                        (session_id,),
                    )
                    row = cursor.fetchone()
                    if row and row.get("data"):
                        raw_data = row["data"]
                        if isinstance(raw_data, str):
                            raw_data = json.loads(raw_data)
                        if isinstance(raw_data, list):
                            return pd.DataFrame(raw_data)
        except Exception as exc:
            errors.append(f"Direct DB: {exc}")

        err_summary = "\n  - ".join(errors)
        raise RuntimeError(f"Could not load data for session {session_id}. Details:\n  - {err_summary}")

    raise ValueError("Either --session or --file must be specified")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ground Contacts and Force Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def extract_turns_and_steps(
    df: pd.DataFrame,
    weight_kg: float = 70.0,
    detection_foot: str = "left",
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    """Detect turns, ground contacts, and associate steps around each turn."""

    t_col = "Time" if "Time" in df.columns else ("time" if "time" in df.columns else None)
    if not t_col:
        raise ValueError("Missing 'Time' column in dataset")

    s_t = pd.to_numeric(df[t_col].iloc[:300], errors="coerce").dropna()
    diffs = np.diff(np.sort(s_t.values))
    diffs = diffs[diffs > 0]
    is_seconds = len(diffs) > 0 and np.median(diffs) < 0.5
    if is_seconds:
        df = df.copy()
        df[t_col] = pd.to_numeric(df[t_col], errors="coerce") * 1000.0

    # 1. Detect Turns (matching UI: default 'left' foot = ESP32_Sensor_1)
    turn_calc = StandaloneTurnCalculator(
        min_net_deg=35.0,
        win_ms=450.0,
        grid_ms=2.0,
        pad_ms=400.0,
        q_rise=0.08,
        min_amp_deg=100.0,
        merge_gap_ms=400.0,
    )

    if detection_foot == "both":
        raw_turns = turn_calc.identify(df, group_sensors=True)
    else:
        sensor_target = "ESP32_Sensor_1" if detection_foot == "left" else "ESP32_Sensor_2"
        foot_df = df[df["Name"] == sensor_target] if "Name" in df.columns else df
        raw_turns = turn_calc.identify(foot_df, group_sensors=False)

    turns = []
    for t in raw_turns:
        t_start_s = t["start_time"] / 1000.0
        t_end_s = t["end_time"] / 1000.0
        duration_ms = t["end_time"] - t["start_time"]
        if duration_ms <= 0:
            continue
        turns.append({
            "turn_index": t["index"],
            "start_time_s": t_start_s,
            "end_time_s": t_end_s,
            "duration_ms": duration_ms,
            "angle_deg": t["angle"],
            "direction": "left" if t["angle"] < 0 else "right" if t["angle"] > 0 else "unknown",
        })

    # 2. Detect Steps / Ground Contacts via ML or Cadence Models
    all_contacts: List[Dict[str, Any]] = []
    rows = df.to_dict(orient="records")

    def _extract_from_viz(viz_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        extracted = []
        for sensor, foot in (("ESP32_Sensor_1", "left"), ("ESP32_Sensor_2", "right")):
            foot_viz = viz_dict.get(sensor, {})
            events = foot_viz.get("contact_events", [])
            times_s = np.asarray(foot_viz.get("t", [])) / 1000.0 if "t" in foot_viz else np.array([])
            acz = np.asarray(foot_viz.get("acz", []))
            for event in events:
                c_start = float(event["timestep_s"])
                c_dur = float(event["contact_time_s"])
                c_end = c_start + c_dur
                pad = 0.1 * c_dur
                w_start, w_end = c_start - pad, c_end + pad
                if times_s.size and acz.size:
                    mask = (times_s >= w_start) & (times_s <= w_end)
                    peak_abs_acz = float(np.max(np.abs(acz[mask]))) if np.any(mask) else 0.0
                else:
                    peak_abs_acz = 0.0

                imu_peak_force_n = float(weight_kg * (peak_abs_acz / 10.0) * 9.80665)
                imu_peak_force_bw = peak_abs_acz / 10.0
                extracted.append({
                    "foot": foot,
                    "start_time_s": c_start,
                    "end_time_s": c_end,
                    "duration_ms": c_dur * 1000.0,
                    "peak_force_n": round(imu_peak_force_n, 1),
                    "peak_force_bw": round(imu_peak_force_bw, 2),
                    "confidence": event.get("confidence"),
                })
        return extracted

    # Attempt 1: StepCadenceCalculator
    try:
        from app.services.calculators.step_cadence_calculator import get_walk_cadence_calculator
        c_calc = get_walk_cadence_calculator()
        c_calc.calculate(rows)
        if getattr(c_calc, "_viz_data", None):
            all_contacts = _extract_from_viz(c_calc._viz_data)
    except Exception:
        pass

    # Attempt 2: WalkGCCalculator directly
    if not all_contacts:
        try:
            from app.services.calculators.walk_gc_calculator import get_walk_gc_calculator
            w_calc = get_walk_gc_calculator()
            w_calc.calculate(rows)
            if getattr(w_calc, "_viz_data", None):
                all_contacts = _extract_from_viz(w_calc._viz_data)
        except Exception:
            pass

    # Attempt 3: MLSprintCalculator / TkeoCadenceCalculator
    if not all_contacts:
        try:
            from app.services.calculators.ml_sprint_calculator import MLSprintCalculator
            s_calc = MLSprintCalculator()
            s_calc.calculate(rows)
            if getattr(s_calc, "_viz_data", None):
                all_contacts = _extract_from_viz(s_calc._viz_data)
        except Exception:
            pass

    if not all_contacts:
        try:
            from app.services.calculators.tkeo_cadence_calculator import TkeoCadenceCalculator
            t_calc = TkeoCadenceCalculator()
            t_calc.calculate(rows)
            if getattr(t_calc, "_viz_data", None):
                all_contacts = _extract_from_viz(t_calc._viz_data)
        except Exception:
            pass

    # Attempt 4: StepDetectorTTest
    if not all_contacts:
        try:
            from app.services.calculators.step_detector_ttest import StepDetectorTTest
            steps = StepDetectorTTest().calculate(df)
            for row in steps.itertuples(index=False):
                all_contacts.append({
                    "foot": str(row.foot),
                    "start_time_s": float(row.t_start),
                    "end_time_s": float(row.t_end),
                    "duration_ms": float(row.contact_ms),
                    "peak_force_n": 0.0,
                    "peak_force_bw": 0.0,
                    "confidence": float(getattr(row, "peak_z", 1.0)),
                })
        except Exception:
            pass

    # Attempt 5: Standalone pure NumPy TKEO fallback
    if not all_contacts:
        sensor_col = "Name" if "Name" in df.columns else None
        for sensor, foot in (("ESP32_Sensor_1", "left"), ("ESP32_Sensor_2", "right")):
            sdf = df[df[sensor_col] == sensor] if sensor_col and sensor_col in df.columns else df
            if sdf.empty:
                continue
            t_arr = pd.to_numeric(sdf[t_col], errors="coerce").to_numpy(dtype=float) / 1000.0
            acz_arr = pd.to_numeric(sdf["AcZ"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if "AcZ" in sdf.columns else np.zeros(len(sdf))
            psi = np.zeros_like(acz_arr)
            if len(acz_arr) >= 3:
                psi[1:-1] = acz_arr[1:-1]**2 - acz_arr[:-2] * acz_arr[2:]
                psi = np.maximum(psi, 0.0)
            win = 15
            smoothed = np.convolve(psi, np.ones(win) / win, mode="same") if len(psi) >= win else psi
            thresh = float(np.percentile(smoothed, 65)) if len(smoothed) else 0.5
            active = smoothed > thresh
            diff_m = np.diff(np.concatenate(([0], active.astype(int), [0])))
            starts_idx = np.where(diff_m == 1)[0]
            ends_idx = np.where(diff_m == -1)[0]
            for s_idx, e_idx in zip(starts_idx, ends_idx):
                if e_idx > s_idx and e_idx < len(t_arr):
                    dur_s = t_arr[e_idx - 1] - t_arr[s_idx]
                    if 0.08 <= dur_s <= 0.8:
                        pk_acz = float(np.max(np.abs(acz_arr[s_idx:e_idx])))
                        all_contacts.append({
                            "foot": foot,
                            "start_time_s": float(t_arr[s_idx]),
                            "end_time_s": float(t_arr[e_idx - 1]),
                            "duration_ms": float(dur_s * 1000.0),
                            "peak_force_n": round(float(weight_kg * (pk_acz / 10.0) * 9.80665), 1),
                            "peak_force_bw": round(float(pk_acz / 10.0), 2),
                            "confidence": 0.9,
                        })

    # Sort contacts chronologically
    all_contacts.sort(key=lambda c: c["start_time_s"])

    # Compute step time and Kinematic Support Force (Step Time / GCT)
    bw_n = weight_kg * 9.80665
    for i in range(len(all_contacts)):
        c = all_contacts[i]
        gct_s = c["duration_ms"] / 1000.0
        if i > 0:
            step_time_s = all_contacts[i]["start_time_s"] - all_contacts[i - 1]["start_time_s"]
            step_time_ms = step_time_s * 1000.0
            c["step_time_ms"] = round(step_time_ms, 1)

            # Kinematic Support Force: F_mean = BW * (t_step / t_contact)
            # Peak Force (Morin sine model): F_peak = F_mean * (pi / 2)
            if gct_s > 0:
                ratio = step_time_s / gct_s
                mean_force_bw_kin = ratio
                peak_force_bw_kin = ratio * (np.pi / 2.0)
                c["mean_force_bw_kin"] = round(mean_force_bw_kin, 2)
                c["peak_force_bw_kin"] = round(peak_force_bw_kin, 2)
                c["mean_force_n_kin"] = round(mean_force_bw_kin * bw_n, 1)
                c["peak_force_n_kin"] = round(peak_force_bw_kin * bw_n, 1)
            else:
                c["mean_force_bw_kin"] = None
                c["peak_force_bw_kin"] = None
                c["mean_force_n_kin"] = None
                c["peak_force_n_kin"] = None
        else:
            c["step_time_ms"] = None
            c["mean_force_bw_kin"] = None
            c["peak_force_bw_kin"] = None
            c["mean_force_n_kin"] = None
            c["peak_force_n_kin"] = None

    # 3. For each turn, extract [-3, -2, -1] steps, [turn step], and [+1] step
    turn_records: List[Dict[str, Any]] = []
    STEP_ROLES = ["step_-3", "step_-2", "step_-1", "turn_step", "step_+1"]

    for turn in turns:
        t_start = turn["start_time_s"]
        t_end = turn["end_time_s"]

        pre_steps = [c for c in all_contacts if c["start_time_s"] < t_start]
        turn_steps = [c for c in all_contacts if (c["start_time_s"] <= t_end and c["end_time_s"] >= t_start)]
        post_steps = [c for c in all_contacts if c["start_time_s"] >= t_end]

        selected: Dict[str, Optional[Dict[str, Any]]] = {
            "step_-3": pre_steps[-3] if len(pre_steps) >= 3 else None,
            "step_-2": pre_steps[-2] if len(pre_steps) >= 2 else None,
            "step_-1": pre_steps[-1] if len(pre_steps) >= 1 else None,
            "turn_step": turn_steps[0] if len(turn_steps) >= 1 else None,
            "step_+1": post_steps[0] if len(post_steps) >= 1 else None,
        }

        row_base = {
            "turn_index": turn["turn_index"],
            "turn_start_s": round(turn["start_time_s"], 3),
            "turn_end_s": round(turn["end_time_s"], 3),
            "turn_duration_ms": round(turn["duration_ms"], 1),
            "turn_angle_deg": round(turn["angle_deg"], 1),
            "turn_direction": turn["direction"],
        }

        record = {
            "turn_index": turn["turn_index"],
            "turn_start_s": round(turn["start_time_s"], 3),
            "turn_duration_ms": round(turn["duration_ms"], 1),
            "turn_angle_deg": round(turn["angle_deg"], 1),
        }

        for role in STEP_ROLES:
            st = selected[role]
            prefix = role
            if st is not None:
                record[f"{prefix}_foot"] = st["foot"]
                record[f"{prefix}_gct_ms"] = round(st["duration_ms"], 1)
                record[f"{prefix}_step_time_ms"] = st["step_time_ms"]
                record[f"{prefix}_force_bw"] = st["mean_force_bw_kin"]
                record[f"{prefix}_force_n"] = st["mean_force_n_kin"]
            else:
                record[f"{prefix}_foot"] = None
                record[f"{prefix}_gct_ms"] = None
                record[f"{prefix}_step_time_ms"] = None
                record[f"{prefix}_force_bw"] = None
                record[f"{prefix}_force_n"] = None

        turn_records.append(record)

    df_results = pd.DataFrame(turn_records)

    # 4. Compute Aggregate Statistics across all turns
    stats: Dict[str, Any] = {
        "total_turns": len(turns),
        "total_contacts": len(all_contacts),
        "roles": {},
    }

    for role in STEP_ROLES:
        gct_col = f"{role}_gct_ms"
        st_col = f"{role}_step_time_ms"
        f_bw_col = f"{role}_force_bw"
        f_n_col = f"{role}_force_n"

        def _mean(col):
            return round(float(df_results[col].dropna().mean()), 2) if col in df_results and not df_results[col].dropna().empty else None

        def _std(col):
            return round(float(df_results[col].dropna().std()), 2) if col in df_results and not df_results[col].dropna().empty else None

        stats["roles"][role] = {
            "gct_ms_mean": _mean(gct_col),
            "gct_ms_std": _std(gct_col),
            "step_time_ms_mean": _mean(st_col),
            "step_time_ms_std": _std(st_col),
            "force_bw_mean": _mean(f_bw_col),
            "force_n_mean": _mean(f_n_col),
        }

    return turn_records, df_results, stats


def main():
    parser = argparse.ArgumentParser(description="Extract steps around turns and compute GCT, Step Time, Force")
    parser.add_argument("--session", type=int, default=6546, help="Session ID (default: 6546)")
    parser.add_argument("--file", type=str, default=None, help="Path to parquet or csv file")
    parser.add_argument("--weight", type=float, default=70.0, help="Subject weight in kg (default: 70.0)")
    parser.add_argument("--detection-foot", type=str, default="left", choices=["left", "right", "both"], help="Foot sensor for turn detection (default: left, matching UI)")
    parser.add_argument("--output-csv", type=str, default="turn_steps_session_6546.csv", help="Output CSV path")
    parser.add_argument("--output-json", type=str, default="turn_steps_summary_6546.json", help="Output summary JSON path")

    args = parser.parse_args()

    print(f"Loading session data (session={args.session}, file={args.file})...")
    df = load_session_df(args.session, args.file)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    print(f"Running turn detection (foot={args.detection_foot}), step detection, and force analysis...")
    turn_records, df_results, stats = extract_turns_and_steps(
        df,
        weight_kg=args.weight,
        detection_foot=args.detection_foot,
    )

    print(f"\n--- Результаты анализа сессии {args.session} ({stats['total_turns']} поворотов) ---")
    print(f"{'Шаг (фаза)':<22} | {'GCT (мс)':<16} | {'Step Time (мс)':<18} | {'Force (BW / N)':<24}")
    print("-" * 86)

    role_names = {
        "step_-3": "3-й шаг до (N-3)",
        "step_-2": "2-й шаг до (N-2)",
        "step_-1": "Предповоротный (N-1)",
        "turn_step": "Шаг разворота (Pivot)",
        "step_+1": "1-й шаг после (N+1)",
    }

    for role, label in role_names.items():
        r = stats["roles"][role]
        gct_str = f"{r['gct_ms_mean']} ± {r['gct_ms_std']}" if r['gct_ms_mean'] is not None else "—"
        st_str = f"{r['step_time_ms_mean']} ± {r['step_time_ms_std']}" if r['step_time_ms_mean'] is not None else "—"
        f_str = f"{r['force_bw_mean']} BW ({r['force_n_mean']} Н)" if r['force_bw_mean'] is not None else "—"
        print(f"{label:<22} | {gct_str:<16} | {st_str:<18} | {f_str:<24}")

    # Save outputs
    df_results.to_csv(args.output_csv, index=False)
    print(f"\nДетальная таблица по всем {len(turn_records)} поворотам сохранена в: {args.output_csv}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Сводка JSON сохранена в: {args.output_json}")


if __name__ == "__main__":
    main()
