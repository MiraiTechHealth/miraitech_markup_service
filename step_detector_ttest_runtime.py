"""Self-contained StepDetectorTTest runtime for the markup service.

The production markup application keeps this detector locally so the
``step-detector-ttest`` calculator does not depend on the sibling backend
checkout containing the development-only module of the same name.

The detection pipeline is intentionally kept in sync with
``MiraiTech-backend:dev/app/services/calculators/step_detector_ttest.py``:
each foot is normalised independently, pressure peaks are detected first,
and contact bounds are recovered by walking from each peak to the local
normalised threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


LEFT_FOOT = "left"
RIGHT_FOOT = "right"
FOOT_NAMES = {
    "ESP32_Sensor_1": LEFT_FOOT,
    "ESP32_Sensor_2": RIGHT_FOOT,
}

STEPS_COLUMNS = [
    "foot",
    "t_start",
    "t_end",
    "t_peak",
    "contact_ms",
    "peak_raw",
    "peak_z",
    "kind",
    "step_num",
    "stride_ms",
]


class StepDetectorTTest:
    """Detect per-foot ground contacts in T-drill IMU sessions."""

    def __init__(
        self,
        lp_cut_hz: float = 10.0,
        baseline_win_s: float = 2.0,
        q_low: float = 0.10,
        q_high: float = 0.90,
        range_floor_frac: float = 0.25,
        min_peak_z: float = 0.45,
        peak_prominence: float = 0.12,
        min_sep_ms: float = 120.0,
        bound_thr: float = 0.30,
        min_contact_ms: float = 80.0,
        max_step_contact_ms: float = 1000.0,
    ):
        self.lp_cut_hz = lp_cut_hz
        self.baseline_win_s = baseline_win_s
        self.q_low = q_low
        self.q_high = q_high
        self.range_floor_frac = range_floor_frac
        self.min_peak_z = min_peak_z
        self.peak_prominence = peak_prominence
        self.min_sep_ms = min_sep_ms
        self.bound_thr = bound_thr
        self.min_contact_ms = min_contact_ms
        self.max_step_contact_ms = max_step_contact_ms

        self._viz_data: Dict[str, Dict[str, Any]] = {}
        self._last_steps: Optional[pd.DataFrame] = None

    def calculate(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """Return detected contacts for both feet, ordered by start time."""
        self._viz_data = {}
        all_steps = []

        for name, group in session_df.groupby("Name"):
            foot = FOOT_NAMES.get(name)
            if foot is None:
                continue

            group = group.sort_values("Time")
            time_s = group["Time"].to_numpy(dtype=float) / 1000.0
            sensor_sum = (
                group["Sensor_1"] + group["Sensor_2"]
            ).to_numpy(dtype=float)
            steps, debug = self._detect_foot(time_s, sensor_sum)
            steps["foot"] = foot
            debug.update({"time_s": time_s, "raw": sensor_sum})
            self._viz_data[foot] = debug
            all_steps.append(steps)

        if not all_steps:
            self._last_steps = pd.DataFrame(columns=STEPS_COLUMNS)
            return self._last_steps

        steps = (
            pd.concat(all_steps, ignore_index=True)
            .sort_values("t_start")
            .reset_index(drop=True)
        )

        is_step = steps["kind"] == "step"
        steps["step_num"] = pd.array([pd.NA] * len(steps), dtype="Int64")
        steps.loc[is_step, "step_num"] = (
            steps[is_step].groupby("foot").cumcount() + 1
        )
        steps["stride_ms"] = np.nan
        steps.loc[is_step, "stride_ms"] = (
            steps[is_step].groupby("foot")["t_start"].diff() * 1000
        ).round(1)

        self._last_steps = steps[STEPS_COLUMNS]
        return self._last_steps

    def _detect_foot(self, time_s: np.ndarray, sensor_sum: np.ndarray):
        """Detect contacts for one foot and return data plus debug arrays."""
        fs = 1.0 / float(np.median(np.diff(time_s)))

        b, a = butter(2, self.lp_cut_hz / (fs / 2), btype="low")
        xf = filtfilt(b, a, sensor_sum)

        win = int(self.baseline_win_s * fs)
        lo = self._rolling_percentile(xf, win, self.q_low)
        hi = self._rolling_percentile(xf, win, self.q_high)
        global_range = np.nanpercentile(xf, 95) - np.nanpercentile(xf, 5)
        value_range = np.maximum(
            hi - lo,
            self.range_floor_frac * global_range,
        )
        z = (xf - lo) / value_range

        peaks, _ = find_peaks(
            z,
            height=self.min_peak_z,
            prominence=self.peak_prominence,
            distance=max(1, int(self.min_sep_ms / 1000 * fs)),
        )

        rows = []
        sample_count = len(z)
        for peak_index, peak in enumerate(peaks):
            left_limit = peaks[peak_index - 1] if peak_index > 0 else 0
            start_index = peak
            while start_index > left_limit and z[start_index] > self.bound_thr:
                start_index -= 1
            if (
                peak_index > 0
                and start_index == left_limit
                and z[start_index] > self.bound_thr
            ):
                start_index = left_limit + int(
                    np.argmin(z[left_limit : peak + 1])
                )

            right_limit = (
                peaks[peak_index + 1]
                if peak_index < len(peaks) - 1
                else sample_count - 1
            )
            end_index = peak
            while end_index < right_limit and z[end_index] > self.bound_thr:
                end_index += 1
            if (
                peak_index < len(peaks) - 1
                and end_index == right_limit
                and z[end_index] > self.bound_thr
            ):
                end_index = peak + int(np.argmin(z[peak : right_limit + 1]))

            duration_ms = (time_s[end_index] - time_s[start_index]) * 1000
            if duration_ms < self.min_contact_ms:
                continue

            if start_index == 0 or end_index >= sample_count - 1:
                kind = "edge"
            elif duration_ms > self.max_step_contact_ms:
                kind = "plateau"
            else:
                kind = "step"

            rows.append(
                {
                    "t_start": time_s[start_index],
                    "t_end": time_s[end_index],
                    "t_peak": time_s[peak],
                    "contact_ms": round(duration_ms, 1),
                    "peak_raw": float(xf[peak]),
                    "peak_z": round(float(z[peak]), 2),
                    "kind": kind,
                }
            )

        steps = pd.DataFrame(
            rows,
            columns=[
                "t_start",
                "t_end",
                "t_peak",
                "contact_ms",
                "peak_raw",
                "peak_z",
                "kind",
            ],
        )
        debug = {"xf": xf, "z": z, "lo": lo, "hi": hi, "peaks": peaks}
        return steps, debug

    @staticmethod
    def _rolling_percentile(x: np.ndarray, win: int, q: float) -> np.ndarray:
        series = pd.Series(x)
        return (
            series.rolling(
                win,
                center=True,
                min_periods=max(10, win // 10),
            )
            .quantile(q)
            .to_numpy()
        )
