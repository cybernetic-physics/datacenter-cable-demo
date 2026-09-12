"""Resolve metric ArUco observations into pelvis-frame wrist targets."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from .arm import rotation_from_rpy_degrees, rpy_degrees_from_rotation
from .models import MarkerOffset

DEFAULT_TARGETING_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "aruco-targeting.yaml"
)


class MarkerObservationSource(Protocol):
    def marker_observation(
        self, marker_id: int
    ) -> tuple[dict[str, object], dict[str, object] | None, float | None]: ...

    def latest(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class TargetingConfig:
    base_from_camera: np.ndarray
    default_offset: MarkerOffset
    max_detection_age_s: float
    max_reprojection_error_px: float
    max_waist_error_rad: float
    camera_serial_number: str
    transform_source: str


@dataclass(frozen=True)
class ResolvedArucoTarget:
    marker_id: int
    detection_age_s: float
    reprojection_error_px: float
    camera_from_marker: np.ndarray
    base_from_marker: np.ndarray
    marker_from_wrist: np.ndarray
    base_from_wrist: np.ndarray


@dataclass(frozen=True)
class VisualizedMarker:
    marker_id: int
    size_m: float
    detection_age_s: float
    saved_for_s: float
    reprojection_error_px: float
    base_from_marker: np.ndarray


@dataclass(frozen=True)
class SavedMarker:
    marker_id: int
    size_m: float
    detection_age_s: float
    reprojection_error_px: float
    camera_from_marker: np.ndarray
    base_from_marker: np.ndarray
    saved_at: float


def pose_transform(xyz: tuple[float, float, float], rpy_deg: tuple[float, float, float]) -> np.ndarray:
    values = np.asarray((*xyz, *rpy_deg), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("pose values must be finite")
    transform = np.eye(4)
    transform[:3, :3] = rotation_from_rpy_degrees(*rpy_deg)
    transform[:3, 3] = xyz
    return transform


def camera_from_marker(rvec: object, tvec_m: object) -> np.ndarray:
    rotation_vector = np.asarray(rvec, dtype=float)
    translation = np.asarray(tvec_m, dtype=float)
    if rotation_vector.shape != (3,) or translation.shape != (3,):
        raise ValueError("marker pose must contain 3-element rvec and tvec_m")
    if not np.all(np.isfinite((rotation_vector, translation))):
        raise ValueError("marker pose must be finite")
    if translation[2] <= 0:
        raise ValueError("marker pose must be in front of the camera")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def compose_target(
    base_from_camera: np.ndarray,
    camera_from_marker_transform: np.ndarray,
    offset: MarkerOffset,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(base_from_marker, base_from_wrist)``."""
    base_camera = _validated_transform(base_from_camera, "base_from_camera")
    camera_marker = _validated_transform(
        camera_from_marker_transform, "camera_from_marker"
    )
    offset.validate()
    marker_wrist = pose_transform(offset.xyz_m, offset.rpy_deg)
    base_marker = base_camera @ camera_marker
    return base_marker, base_marker @ marker_wrist


def _validated_transform(value: object, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-9):
        raise ValueError(f"{name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name} rotation must be orthonormal")
    return transform.copy()


def _pose_json(transform: np.ndarray) -> dict[str, list[float]]:
    return {
        "xyz": np.round(transform[:3, 3], 6).tolist(),
        "rpy_deg": np.round(
            rpy_degrees_from_rotation(transform[:3, :3]), 4
        ).tolist(),
    }


class ArucoTargeting:
    """Latch valid marker poses for the session and resolve wrist targets."""

    def __init__(
        self,
        observations: MarkerObservationSource,
        config_path: Path = DEFAULT_TARGETING_CONFIG,
    ):
        self.observations = observations
        self.config = self._load_config(config_path)
        self._saved_lock = threading.Lock()
        self._saved_markers: dict[int, SavedMarker] = {}

    @staticmethod
    def _load_config(path: Path) -> TargetingConfig:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))["aruco_targeting"]
            transform = _validated_transform(
                payload["base_from_internal_camera"]["matrix"], "base_from_internal_camera"
            )
            offset_body = payload["default_offset"]
            offset = MarkerOffset(
                tuple(float(v) for v in offset_body["xyz_m"]),
                tuple(float(v) for v in offset_body["rpy_deg"]),
            )
            offset.validate()
            config = TargetingConfig(
                base_from_camera=transform,
                default_offset=offset,
                max_detection_age_s=float(payload["max_detection_age_s"]),
                max_reprojection_error_px=float(payload["max_reprojection_error_px"]),
                max_waist_error_rad=float(payload["max_waist_error_rad"]),
                camera_serial_number=str(payload["camera_serial_number"]),
                transform_source=str(payload["base_from_internal_camera"]["source"]),
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise RuntimeError(f"invalid ArUco targeting configuration: {path}") from error
        limits = (
            config.max_detection_age_s,
            config.max_reprojection_error_px,
            config.max_waist_error_rad,
        )
        if not np.all(np.isfinite(limits)) or any(value <= 0 for value in limits):
            raise RuntimeError(f"invalid ArUco targeting configuration: {path}")
        return config

    def configuration(self) -> dict[str, object]:
        return {
            "base_frame": "g1_pelvis",
            "camera_frame": "internal_d435i_color_optical",
            "camera_serial_number": self.config.camera_serial_number,
            "transform_source": self.config.transform_source,
            "base_from_camera": _pose_json(self.config.base_from_camera),
            "default_offset": {
                "xyz_m": list(self.config.default_offset.xyz_m),
                "rpy_deg": list(self.config.default_offset.rpy_deg),
            },
            "max_detection_age_s": self.config.max_detection_age_s,
            "max_reprojection_error_px": self.config.max_reprojection_error_px,
            "max_waist_error_rad": self.config.max_waist_error_rad,
        }

    def resolve(
        self, marker_id: int, offset: MarkerOffset | None = None
    ) -> ResolvedArucoTarget:
        if marker_id < 0:
            raise ValueError("marker_id must be non-negative")
        saved = self._saved_marker(marker_id)
        if saved is None:
            result, marker, age_s = self.observations.marker_observation(marker_id)
            age_s = self._validate_frame(result, age_s)
            if marker is None:
                raise RuntimeError(f"ArUco marker {marker_id} is not detected")
            candidate = self._validated_marker(result, marker, age_s)
            if candidate.marker_id != marker_id:
                raise RuntimeError(f"ArUco marker {marker_id} is not detected")
            saved = self._save_first(candidate)
        selected_offset = offset or self.config.default_offset
        marker_wrist = pose_transform(selected_offset.xyz_m, selected_offset.rpy_deg)
        return ResolvedArucoTarget(
            marker_id=marker_id,
            detection_age_s=saved.detection_age_s,
            reprojection_error_px=saved.reprojection_error_px,
            camera_from_marker=saved.camera_from_marker.copy(),
            base_from_marker=saved.base_from_marker.copy(),
            marker_from_wrist=marker_wrist,
            base_from_wrist=saved.base_from_marker @ marker_wrist,
        )

    def _validate_frame(
        self, result: dict[str, object], age_s: float | None
    ) -> float:
        if result.get("observed_at") is None or age_s is None:
            raise RuntimeError("no ArUco detection frame is available")
        try:
            age = float(age_s)
        except (TypeError, ValueError) as error:
            raise RuntimeError("ArUco detection age is invalid") from error
        if not np.isfinite(age) or age < 0:
            raise RuntimeError("ArUco detection age is invalid")
        if age > self.config.max_detection_age_s:
            raise RuntimeError(
                f"ArUco detection is stale ({age:.3f} s; "
                f"limit {self.config.max_detection_age_s:.3f} s)"
            )
        if not result.get("calibration_valid"):
            raise RuntimeError("internal-camera metric calibration is invalid")
        if str(result.get("camera_serial_number")) != self.config.camera_serial_number:
            raise RuntimeError("camera identity does not match the targeting transform")
        return age

    def _validated_marker(
        self, result: dict[str, object], marker: dict[str, object], age_s: float
    ) -> SavedMarker:
        rvec, tvec = marker.get("rvec"), marker.get("tvec_m")
        reprojection_error = marker.get("reprojection_error_px")
        if rvec is None or tvec is None or reprojection_error is None:
            raise RuntimeError(
                "metric marker pose is unavailable; configure the physical marker edge length"
            )
        try:
            error_px = float(reprojection_error)
        except (TypeError, ValueError) as error:
            raise RuntimeError("marker reprojection error is invalid") from error
        if not np.isfinite(error_px):
            raise RuntimeError("marker reprojection error is invalid")
        if error_px > self.config.max_reprojection_error_px:
            raise RuntimeError(
                f"marker reprojection error is too high ({error_px:.3f} px; "
                f"limit {self.config.max_reprojection_error_px:.3f} px)"
            )
        camera_marker = camera_from_marker(rvec, tvec)
        try:
            size_m = float(result["marker_length_mm"]) / 1000.0
            marker_id = int(marker["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("metric marker size or ID is invalid") from error
        if not np.isfinite(size_m) or size_m <= 0:
            raise RuntimeError("metric marker size or ID is invalid")
        return SavedMarker(
            marker_id=marker_id,
            size_m=size_m,
            detection_age_s=age_s,
            reprojection_error_px=error_px,
            camera_from_marker=camera_marker,
            base_from_marker=self.config.base_from_camera @ camera_marker,
            saved_at=time.monotonic(),
        )

    def _saved_marker(self, marker_id: int) -> SavedMarker | None:
        with self._saved_lock:
            return self._saved_markers.get(marker_id)

    def _save_first(self, marker: SavedMarker) -> SavedMarker:
        with self._saved_lock:
            return self._saved_markers.setdefault(marker.marker_id, marker)

    def visualized_markers(self) -> tuple[VisualizedMarker, ...]:
        """Latch new valid markers and return all poses saved this session."""
        result = self.observations.latest()
        age_value = result.get("age_s")
        try:
            age_s = float(age_value)
        except (TypeError, ValueError):
            age_s = float("nan")
        try:
            age_s = self._validate_frame(result, age_s)
        except RuntimeError:
            pass
        else:
            items = result.get("markers", [])
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    try:
                        marker_id = int(item["id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if self._saved_marker(marker_id) is not None:
                        continue
                    try:
                        self._save_first(self._validated_marker(result, item, age_s))
                    except RuntimeError:
                        continue
        now = time.monotonic()
        with self._saved_lock:
            saved = tuple(self._saved_markers.values())
        return tuple(
            VisualizedMarker(
                marker.marker_id,
                marker.size_m,
                marker.detection_age_s,
                max(0.0, now - marker.saved_at),
                marker.reprojection_error_px,
                marker.base_from_marker.copy(),
            )
            for marker in saved
        )

    @staticmethod
    def resolution_json(resolved: ResolvedArucoTarget) -> dict[str, object]:
        return {
            "marker_id": resolved.marker_id,
            "marker_pose_source": "session_latch",
            "detection_age_s": round(resolved.detection_age_s, 4),
            "reprojection_error_px": round(resolved.reprojection_error_px, 4),
            "camera_marker": _pose_json(resolved.camera_from_marker),
            "base_marker": _pose_json(resolved.base_from_marker),
            "marker_wrist_offset": _pose_json(resolved.marker_from_wrist),
            "base_wrist_target": _pose_json(resolved.base_from_wrist),
        }
