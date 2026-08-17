#!/usr/bin/env python3
"""
Extract 3 steps before turn, 1 step after turn, and calculate GCT, Step Time, and Force.

Usage:
    python extract_turn_steps.py --session 6546 [--weight 70.0] [--output-csv turn_steps_session_6546.csv]
    python extract_turn_steps.py --file session_6546.parquet [--weight 70.0]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Set up paths to import MiraiTech backend
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BACKEND_ROOT = SCRIPT_DIR.parent / "MiraiTech-backend"
BACKEND_ROOT = Path(os.environ.get("MIRAITECH_BACKEND_ROOT", str(DEFAULT_BACKEND_ROOT))).expanduser().resolve()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_session_df(session_id: Optional[int], file_path: Optional[str]) -> pd.DataFrame:
    """Load session dataframe from file or storage."""
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
        from app.services.session_data_storage import get_session_parquet_bytes, load_session_dataframe

        try:
            df = load_session_dataframe(session_id)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        raw_bytes = get_session_parquet_bytes(session_id)
        if raw_bytes is None:
            raise RuntimeError(f"Could not load data for session {session_id} from storage")
        return pd.read_parquet(io.BytesIO(raw_bytes))

    raise ValueError("Either --session or --file must be specified")


def extract_turns_and_steps(
    df: pd.DataFrame,
    weight_kg: float = 70.0,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    """Detect turns and steps, then match steps around each turn."""
    from app.services.calculators.turn_calculator import TurnCalculator
    from app.services.calculators.step_cadence_calculator import get_walk_cadence_calculator
    from app.services.calculators.ground_contact_force_calculator import GroundContactForceCalculator

    # Normalise Time to milliseconds
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

    # 1. Detect Turns (Yo-Yo / Beep Test detector)
    turn_calc = TurnCalculator(min_change_deg=100, max_duration_ms=2000)
    raw_turns = turn_calc.identify(df, group_sensors=True)

    # Convert turn timestamps to seconds
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

    # 2. Detect Ground Contacts (Walk/Run BiLSTM cadence model)
    cadence_calc = get_walk_cadence_calculator()
    rows = df.to_dict(orient="records")
    cadence_calc.calculate(rows)

    viz = getattr(cadence_calc, "_viz_data", {})
    all_contacts: List[Dict[str, Any]] = []

    force_calc = GroundContactForceCalculator()

    for sensor, foot in (("ESP32_Sensor_1", "left"), ("ESP32_Sensor_2", "right")):
        foot_viz = viz.get(sensor, {})
        events = foot_viz.get("contact_events", [])
        times_s = np.asarray(foot_viz.get("t", [])) / 1000.0 if "t" in foot_viz else np.array([])
        acz = np.asarray(foot_viz.get("acz", []))

        # Sensor data slice for pressure sum
        sensor_foot_df = df[df["Name"] == sensor] if "Name" in df.columns else df
        sensor_t = pd.to_numeric(sensor_foot_df[t_col], errors="coerce").to_numpy(dtype=float) / 1000.0
        p_cols = [c for c in ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4"] if c in sensor_foot_df.columns]
        p_data = sensor_foot_df[p_cols].to_numpy(dtype=float) if p_cols else np.zeros((len(sensor_foot_df), 1))
        p_total = p_data.sum(axis=1) if len(p_data) else np.zeros(len(sensor_foot_df))

        for event in events:
            c_start = float(event["timestep_s"])
            c_dur = float(event["contact_time_s"])
            c_end = c_start + c_dur

            # Force calculation from AcZ in extended contact window
            w_start, w_end = force_calc.extended_contact_window(c_start, c_dur)
            peak_acz = force_calc.peak_abs_acz_in_window(times_s, acz, w_start, w_end)
            peak_force_n = force_calc.peak_force_n_from_abs_acz(peak_abs_acz=peak_acz, weight_kg=weight_kg)
            peak_force_bw = peak_force_n / (weight_kg * 9.80665) if weight_kg > 0 else 0.0

            # Mean force over contact window
            if times_s.size and acz.size:
                mask = (times_s >= c_start) & (times_s <= c_end)
                if np.any(mask):
                    mean_acz = float(np.mean(np.abs(acz[mask])))
                    mean_force_n = force_calc.peak_force_n_from_abs_acz(mean_acz, weight_kg)
                    mean_force_bw = mean_force_n / (weight_kg * 9.80665) if weight_kg > 0 else 0.0
                else:
                    mean_force_n = peak_force_n * 0.6
                    mean_force_bw = peak_force_bw * 0.6
            else:
                mean_force_n = peak_force_n * 0.6
                mean_force_bw = peak_force_bw * 0.6

            # Sensor pressure sum
            if sensor_t.size and p_total.size:
                p_mask = (sensor_t >= c_start) & (sensor_t <= c_end)
                mean_pressure = float(np.mean(p_total[p_mask])) if np.any(p_mask) else 0.0
            else:
                mean_pressure = 0.0

            all_contacts.append({
                "foot": foot,
                "start_time_s": c_start,
                "end_time_s": c_end,
                "duration_ms": c_dur * 1000.0,
                "peak_force_n": round(peak_force_n, 1),
                "peak_force_bw": round(peak_force_bw, 2),
                "mean_force_n": round(mean_force_n, 1),
                "mean_force_bw": round(mean_force_bw, 2),
                "mean_pressure": round(mean_pressure, 1),
                "confidence": event.get("confidence"),
            })

    # Sort all contacts chronologically
    all_contacts.sort(key=lambda c: c["start_time_s"])

    # Compute step time (inter-contact onset time) and Kinematic Support Force (Step Time / GCT)
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

        # Steps strictly before turn start
        pre_steps = [c for c in all_contacts if c["start_time_s"] < t_start]
        # Step during turn (overlapping with [t_start, t_end])
        turn_steps = [
            c for c in all_contacts
            if (c["start_time_s"] <= t_end and c["end_time_s"] >= t_start)
        ]
        # Steps after turn end
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

        # Flat record for CSV export
        record = dict(row_base)
        for role in STEP_ROLES:
            st = selected[role]
            prefix = role
            if st is not None:
                record[f"{prefix}_foot"] = st["foot"]
                record[f"{prefix}_time_s"] = round(st["start_time_s"], 3)
                record[f"{prefix}_gct_ms"] = round(st["duration_ms"], 1)
                record[f"{prefix}_step_time_ms"] = st["step_time_ms"]
                # Kinematic Support Force (Step Time / GCT)
                record[f"{prefix}_support_force_bw"] = st["mean_force_bw_kin"]
                record[f"{prefix}_support_force_n"] = st["mean_force_n_kin"]
                record[f"{prefix}_peak_force_bw_kin"] = st["peak_force_bw_kin"]
                record[f"{prefix}_peak_force_n_kin"] = st["peak_force_n_kin"]
                # IMU AcZ Force
                record[f"{prefix}_imu_peak_force_n"] = st["peak_force_n"]
                record[f"{prefix}_imu_peak_force_bw"] = st["peak_force_bw"]
            else:
                record[f"{prefix}_foot"] = None
                record[f"{prefix}_time_s"] = None
                record[f"{prefix}_gct_ms"] = None
                record[f"{prefix}_step_time_ms"] = None
                record[f"{prefix}_support_force_bw"] = None
                record[f"{prefix}_support_force_n"] = None
                record[f"{prefix}_peak_force_bw_kin"] = None
                record[f"{prefix}_peak_force_n_kin"] = None
                record[f"{prefix}_imu_peak_force_n"] = None
                record[f"{prefix}_imu_peak_force_bw"] = None

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
        sup_bw_col = f"{role}_support_force_bw"
        sup_n_col = f"{role}_support_force_n"
        pk_kin_bw_col = f"{role}_peak_force_bw_kin"
        pk_kin_n_col = f"{role}_peak_force_n_kin"
        imu_n_col = f"{role}_imu_peak_force_n"
        imu_bw_col = f"{role}_imu_peak_force_bw"

        def _mean(col):
            return round(float(df_results[col].dropna().mean()), 2) if col in df_results and not df_results[col].dropna().empty else None

        def _std(col):
            return round(float(df_results[col].dropna().std()), 2) if col in df_results and not df_results[col].dropna().empty else None

        stats["roles"][role] = {
            "gct_ms_mean": _mean(gct_col),
            "gct_ms_std": _std(gct_col),
            "step_time_ms_mean": _mean(st_col),
            "step_time_ms_std": _std(st_col),
            "support_force_bw_mean": _mean(sup_bw_col),
            "support_force_n_mean": _mean(sup_n_col),
            "peak_force_kin_bw_mean": _mean(pk_kin_bw_col),
            "peak_force_kin_n_mean": _mean(pk_kin_n_col),
            "imu_peak_force_n_mean": _mean(imu_n_col),
            "imu_peak_force_bw_mean": _mean(imu_bw_col),
        }

    return turn_records, df_results, stats


def main():
    parser = argparse.ArgumentParser(description="Extract steps around turns and compute GCT, Step Time, Force")
    parser.add_argument("--session", type=int, default=6546, help="Session ID (default: 6546)")
    parser.add_argument("--file", type=str, default=None, help="Path to parquet or csv file")
    parser.add_argument("--weight", type=float, default=70.0, help="Subject weight in kg (default: 70.0)")
    parser.add_argument("--output-csv", type=str, default="turn_steps_session_6546.csv", help="Output CSV path")
    parser.add_argument("--output-json", type=str, default="turn_steps_summary_6546.json", help="Output summary JSON path")

    args = parser.parse_args()

    print(f"Loading session data (session={args.session}, file={args.file})...")
    df = load_session_df(args.session, args.file)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    print("Running turn detection, step detection, and force analysis...")
    turn_records, df_results, stats = extract_turns_and_steps(df, weight_kg=args.weight)

    print(f"\n--- Результаты анализа сессии {args.session} ({stats['total_turns']} поворотов) ---")
    print(f"{'Шаг (фаза)':<22} | {'GCT (мс)':<16} | {'Step Time (мс)':<18} | {'Support Force (BW / N)':<24} | {'Peak Force Kin (BW)':<20} | {'IMU AcZ Peak (BW)':<18}")
    print("-" * 128)

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
        sup_str = f"{r['support_force_bw_mean']} BW ({r['support_force_n_mean']} Н)" if r['support_force_bw_mean'] is not None else "—"
        pk_kin_str = f"{r['peak_force_kin_bw_mean']} BW ({r['peak_force_kin_n_mean']} Н)" if r['peak_force_kin_bw_mean'] is not None else "—"
        imu_str = f"{r['imu_peak_force_bw_mean']} BW ({r['imu_peak_force_n_mean']} Н)" if r['imu_peak_force_bw_mean'] is not None else "—"
        print(f"{label:<22} | {gct_str:<16} | {st_str:<18} | {sup_str:<24} | {pk_kin_str:<20} | {imu_str:<18}")

    # Save outputs
    df_results.to_csv(args.output_csv, index=False)
    print(f"\nДетальная таблица по всем {len(turn_records)} поворотам сохранена в: {args.output_csv}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Сводка JSON сохранена в: {args.output_json}")


if __name__ == "__main__":
    main()
