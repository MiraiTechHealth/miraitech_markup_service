"""JumpBiLSTM runtime used by the local markup calculator API.

The architecture and preprocessing mirror backend commit ``842b162``, where
the 500 Hz JumpBiLSTM model was introduced.  Metric aggregation is inherited
from the backend calculator; this module replaces only model loading,
preprocessing and inference.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from scipy.ndimage import median_filter

from app.services.calculators.ml_jump_metrics_calculator import (
    MLJumpMetricsCalculator as BackendJumpMetricsCalculator,
)
from app.services.calculators.step_calculator import tkeo, unwrap_angle_degrees


MODEL_DIR = Path(__file__).resolve().parent / "models" / "jump_bilstm"
TRAIN_FS_HZ = 500.0
TKEO_WIN = 15
RESAMPLE_TOLERANCE = 0.1


class JumpBiLSTM(nn.Module):
    """Multi-scale CNN -> BiLSTM -> temporal attention classifier."""

    def __init__(
        self,
        n_features: int = 24,
        hidden_size: int = 128,
        num_layers: int = 2,
        branch_channels: int = 48,
        proj_channels: int = 128,
    ) -> None:
        super().__init__()

        def branch(kernel_size: int) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(n_features, branch_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.Conv1d(branch_channels, branch_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
            )

        self.branch_s = branch(3)
        self.branch_m = branch(9)
        self.branch_l = branch(21)
        self.projection = nn.Sequential(
            nn.Conv1d(branch_channels * 3, proj_channels, kernel_size=1),
            nn.BatchNorm1d(proj_channels),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        lstm_output_size = hidden_size * 2
        self.norm = nn.LayerNorm(lstm_output_size)
        self.attn = nn.Linear(lstm_output_size, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_output_size, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat(
            [self.branch_s(inputs), self.branch_m(inputs), self.branch_l(inputs)],
            dim=1,
        )
        sequence = self.projection(multi_scale).permute(0, 2, 1)
        sequence, _ = self.lstm(sequence)
        sequence = self.norm(sequence)
        weights = torch.softmax(self.attn(sequence), dim=1)
        sequence = sequence + (weights * sequence).sum(dim=1, keepdim=True)
        return self.classifier(sequence)


def _resample_to_hz(
    times: np.ndarray,
    features: np.ndarray,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(times) < 2 or features.shape[0] != len(times):
        return times, features

    time_deltas = np.diff(times)
    positive_deltas = time_deltas[time_deltas > 0]
    if not len(positive_deltas):
        return times, features

    current_hz = 1000.0 / float(np.median(positive_deltas))
    if abs(current_hz - target_hz) / target_hz <= RESAMPLE_TOLERANCE:
        return times, features

    uniform_times = np.arange(times[0], times[-1], 1000.0 / target_hz)
    if len(uniform_times) < 2:
        return times, features

    resampled = np.column_stack(
        [np.interp(uniform_times, times, features[:, index]) for index in range(features.shape[1])]
    ).astype(np.float32)
    return uniform_times, resampled


class MarkupJumpBiLSTMCalculator(BackendJumpMetricsCalculator):
    """Backend jump metrics with the 24-feature JumpBiLSTM inference stack."""

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        min_contact_time_ms: float = 50.0,
        max_contact_time_ms: float = 10000.0,
        device: Optional[str] = None,
        infer_batch: int = 128,
    ) -> None:
        model_path = Path(model_dir) if model_dir else MODEL_DIR
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        config = joblib.load(model_path / "jump_bilstm_config.pkl")
        self.feature_cols: List[str] = list(config["feature_cols"])
        self.window_size = int(config["window_size"])
        self.n_features = int(config["n_features"])
        self.train_fs_hz = float(config.get("sample_rate_hz", TRAIN_FS_HZ))
        self.tkeo_win = int(config.get("tkeo_win", TKEO_WIN))
        self.scaler = joblib.load(model_path / "jump_bilstm_scaler.pkl")

        self.model = JumpBiLSTM(
            n_features=self.n_features,
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_lstm_layers"]),
            branch_channels=int(config["branch_channels"]),
            proj_channels=int(config["proj_channels"]),
        ).to(self.device)
        state = torch.load(
            model_path / "jump_bilstm.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()

        self.voting_step = int(config.get("recommended_infer_stride", 25))
        self.voting_threshold = float(config.get("recommended_vote_threshold", 0.5))
        self.median_kernel = int(config.get("recommended_median_kernel", 75))
        self.min_jump_samples = int(config.get("recommended_min_jump_samples", 100))
        self.min_contact_time_ms = min_contact_time_ms
        self.max_contact_time_ms = max_contact_time_ms
        self.infer_batch = infer_batch
        self._infer_lock = threading.RLock()
        self.result = None
        self._calibration = None
        self.analysis_details: Dict[str, Dict] = {
            "ESP32_Sensor_1": {},
            "ESP32_Sensor_2": {},
        }

        logger.info(
            "Markup JumpBiLSTM loaded | device={} | window={} @ {}Hz | features={}",
            self.device,
            self.window_size,
            self.train_fs_hz,
            self.n_features,
        )

    def _preprocess_foot_raw(
        self,
        foot_data: List[Dict],
        foot_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_columns = [
            "AcX",
            "AcY",
            "AcZ",
            "XData",
            "YData",
            "ZData",
            "GravityZ",
            "Sensor_1",
            "Sensor_2",
            "Sensor_3",
            "Sensor_4",
        ]
        rows = []
        for entry in foot_data:
            try:
                timestamp = float(entry.get("Time", 0))
                values = [float(entry.get(column, 0) or 0) for column in raw_columns]
                rows.append([timestamp, *values])
            except (TypeError, ValueError):
                continue

        if not rows:
            return (
                np.array([], dtype=np.float64),
                np.empty((0, self.n_features), dtype=np.float32),
            )

        rows.sort(key=lambda row: row[0])
        data = np.asarray(rows, dtype=np.float64)
        times = data[:, 0]
        acx, acy, acz = data[:, 1], data[:, 2], data[:, 3]
        gravity_z = data[:, 7]
        sensor_1, sensor_2, sensor_3, sensor_4 = (
            data[:, 8],
            data[:, 9],
            data[:, 10],
            data[:, 11],
        )
        x_data, y_data, z_data = (
            unwrap_angle_degrees(pd.Series(data[:, index])).to_numpy(dtype=np.float64)
            for index in (4, 5, 6)
        )

        if foot_type == "right":
            acx = -acx
            acy = -acy

        x_data = x_data - x_data[0]
        y_data = y_data - y_data[0]
        z_data = z_data - z_data[0]

        acceleration_magnitude = np.sqrt(acx**2 + acy**2 + acz**2)
        gyro_magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        delta_x = np.diff(x_data, prepend=x_data[0])
        delta_y = np.diff(y_data, prepend=y_data[0])
        delta_z = np.diff(z_data, prepend=z_data[0])
        gyro_delta_magnitude = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        sensor_total = sensor_1 + sensor_2 + sensor_3 + sensor_4

        columns = {
            "AcZ": acz,
            "AcX": acx,
            "AcY": acy,
            "XData": x_data,
            "YData": y_data,
            "ZData": z_data,
            "GravityZ": gravity_z,
            "Sensor_1": sensor_1,
            "Sensor_2": sensor_2,
            "Sensor_3": sensor_3,
            "Sensor_4": sensor_4,
            "acc_mag": acceleration_magnitude,
            "gyro_mag": gyro_magnitude,
            "jerk_AcX": np.diff(acx, prepend=acx[0]),
            "jerk_AcY": np.diff(acy, prepend=acy[0]),
            "jerk_AcZ": np.diff(acz, prepend=acz[0]),
            "d_XData": delta_x,
            "d_YData": delta_y,
            "d_ZData": delta_z,
            "gyro_delta_mag": gyro_delta_magnitude,
            "tkeo_acc": tkeo(acceleration_magnitude, win=self.tkeo_win),
            "tkeo_gyro": tkeo(gyro_delta_magnitude, win=self.tkeo_win),
            "sensor_total": sensor_total,
            "d_sensor_total": np.diff(sensor_total, prepend=sensor_total[0]),
        }

        missing = [column for column in self.feature_cols if column not in columns]
        if missing:
            raise KeyError(f"JumpBiLSTM preprocessing cannot build features: {missing}")

        features = np.column_stack(
            [columns[column] for column in self.feature_cols]
        ).astype(np.float32)
        return times, features

    def _preprocess_foot(
        self,
        foot_data: List[Dict],
        foot_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        times, features = self._preprocess_foot_raw(foot_data, foot_type)
        if not len(times):
            return times, features
        return _resample_to_hz(times, features, self.train_fs_hz)

    @torch.no_grad()
    def _predict_foot(self, features: np.ndarray) -> np.ndarray:
        sample_count = len(features)
        if sample_count < self.window_size:
            return np.zeros(sample_count, dtype=int)

        scaled = self.scaler.transform(features).astype(np.float32)
        offsets = list(
            range(0, sample_count - self.window_size + 1, self.voting_step)
        )
        if (sample_count - self.window_size) % self.voting_step:
            offsets.append(sample_count - self.window_size)

        probability_sums = np.zeros(sample_count, dtype=np.float32)
        vote_counts = np.zeros(sample_count, dtype=np.float32)
        for batch_start in range(0, len(offsets), self.infer_batch):
            batch_offsets = offsets[batch_start : batch_start + self.infer_batch]
            windows = np.stack(
                [scaled[offset : offset + self.window_size] for offset in batch_offsets]
            )
            inputs = torch.from_numpy(windows).permute(0, 2, 1).to(self.device)
            probabilities = torch.softmax(self.model(inputs), dim=2)[..., 1].cpu().numpy()
            for index, offset in enumerate(batch_offsets):
                probability_sums[offset : offset + self.window_size] += probabilities[index]
                vote_counts[offset : offset + self.window_size] += 1.0

        average_probabilities = np.divide(
            probability_sums,
            vote_counts,
            out=np.zeros_like(probability_sums),
            where=vote_counts > 0,
        )
        predictions = (average_probabilities >= self.voting_threshold).astype(int)
        predictions = (
            median_filter(predictions.astype(np.float64), size=self.median_kernel) >= 0.5
        ).astype(int)

        transitions = np.diff(np.concatenate(([0], predictions, [0])))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        for start, end in zip(starts, ends):
            if end - start < self.min_jump_samples:
                predictions[start:end] = 0
        return predictions
