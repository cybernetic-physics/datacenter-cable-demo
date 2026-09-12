"""OpenCV ArUco detection for the internal G1 camera."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from .camera import CameraHub, CameraUnavailable, multipart_jpeg

DICTIONARY_NAMES = tuple(
    [f"DICT_{bits}X{bits}_{size}" for bits in range(4, 8) for size in (50, 100, 250, 1000)]
    + ["DICT_ARUCO_ORIGINAL"]
)
DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "config" / "camera-calibrations.yaml"


class ArucoVision:
    """Detect markers and retain the latest JSON-safe observations."""

    def __init__(self, camera: CameraHub, calibration_path: Path = DEFAULT_CALIBRATION):
        self.camera = camera
        self.calibration = self._load_calibration(calibration_path)
        self._lock = threading.Lock()
        self._dictionary = "DICT_4X4_50"
        self._marker_length_mm: float | None = None
        self._active = False
        self._latest = self._empty_result()
        self._latest_monotonic: float | None = None

    @staticmethod
    def _load_calibration(path: Path) -> dict[str, object]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            calibration = payload["cameras"]["internal"]
            matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid internal camera calibration: {path}") from error
        if (
            matrix.shape != (3, 3)
            or distortion.ndim != 1
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(distortion))
        ):
            raise RuntimeError(f"invalid internal camera calibration: {path}")
        calibration = dict(calibration)
        calibration["camera_matrix"] = matrix
        calibration["distortion_coefficients"] = distortion
        return calibration

    def configure(self, dictionary: str, marker_length_mm: float | None) -> dict[str, object]:
        if dictionary not in DICTIONARY_NAMES:
            raise ValueError(f"unsupported ArUco dictionary: {dictionary}")
        if marker_length_mm is not None and not 0 < marker_length_mm <= 1000:
            raise ValueError("marker_length_mm must be in (0, 1000]")
        with self._lock:
            self._dictionary = dictionary
            self._marker_length_mm = marker_length_mm
            self._latest = self._empty_result(dictionary, marker_length_mm)
            self._latest_monotonic = None
        return self.configuration()

    def configuration(self) -> dict[str, object]:
        calibration = self._calibration_status()
        with self._lock:
            dictionary = self._dictionary
            marker_length_mm = self._marker_length_mm
            active = self._active
        return {
            "dictionaries": list(DICTIONARY_NAMES),
            "dictionary": dictionary,
            "marker_length_mm": marker_length_mm,
            "active": active,
            "calibration": calibration,
        }

    def latest(self) -> dict[str, object]:
        with self._lock:
            result = dict(self._latest)
            observed = self._latest_monotonic
        result["age_s"] = None if observed is None else max(0.0, time.monotonic() - observed)
        return result

    def marker_observation(
        self, marker_id: int
    ) -> tuple[dict[str, object], dict[str, object] | None, float | None]:
        """Return one marker and its local receive age from the latest frame."""
        with self._lock:
            result = dict(self._latest)
            marker = next(
                (dict(item) for item in self._latest["markers"] if item["id"] == marker_id),
                None,
            )
            observed = self._latest_monotonic
        age_s = None if observed is None else max(0.0, time.monotonic() - observed)
        return result, marker, age_s

    def annotated_stream(self) -> Iterator[bytes]:
        frames = self.camera.jpeg_stream("internal")
        with self._lock:
            dictionary_name = self._dictionary
            marker_length_mm = self._marker_length_mm
        calibration_status = self._calibration_status()
        return self._process_stream(
            frames, dictionary_name, marker_length_mm, calibration_status["valid"]
        )

    def _process_stream(
        self,
        frames: Iterator[bytes],
        dictionary_name: str,
        marker_length_mm: float | None,
        calibration_valid: bool,
    ) -> Iterator[bytes]:
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary_id),
            cv2.aruco.DetectorParameters(),
        )
        with self._lock:
            self._active = True
            self._latest = self._empty_result(dictionary_name, marker_length_mm, calibration_valid)
            self._latest_monotonic = None
        try:
            for jpeg in frames:
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                frame_calibration_valid = calibration_valid and (
                    image.shape[1] == int(self.calibration["width"])
                    and image.shape[0] == int(self.calibration["height"])
                )
                corners, ids, _ = detector.detectMarkers(image)
                markers = self._markers(
                    image, corners, ids, marker_length_mm, frame_calibration_valid
                )
                observed_at = datetime.now(timezone.utc).isoformat()
                result = {
                    "observed_at": observed_at,
                    "frame_size": [int(image.shape[1]), int(image.shape[0])],
                    "camera_serial_number": str(self.calibration["serial_number"]),
                    "dictionary": dictionary_name,
                    "marker_length_mm": marker_length_mm,
                    "calibration_valid": frame_calibration_valid,
                    "coordinate_frame": "opencv_camera_x_right_y_down_z_forward",
                    "markers": markers,
                }
                with self._lock:
                    self._latest = result
                    self._latest_monotonic = time.monotonic()
                ok, annotated = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if ok:
                    yield multipart_jpeg(annotated.tobytes())
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()
            with self._lock:
                self._active = False

    def _markers(
        self,
        image: np.ndarray,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
        marker_length_mm: float | None,
        calibration_valid: bool,
    ) -> list[dict[str, object]]:
        if ids is None:
            return []
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        results: list[dict[str, object]] = []
        for marker_id, marker_corners in zip(ids.reshape(-1), corners):
            points = marker_corners.reshape(4, 2).astype(np.float64)
            result: dict[str, object] = {
                "id": int(marker_id),
                "corners_px": points.tolist(),
                "rvec": None,
                "tvec_m": None,
                "reprojection_error_px": None,
            }
            if marker_length_mm is not None and calibration_valid:
                result.update(self._estimate_pose(image, points, marker_length_mm / 1000.0))
            results.append(result)
        return results

    def _estimate_pose(
        self, image: np.ndarray, image_points: np.ndarray, marker_length_m: float
    ) -> dict[str, object]:
        half = marker_length_m / 2.0
        object_points = np.asarray(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        matrix = self.calibration["camera_matrix"]
        distortion = self.calibration["distortion_coefficients"]
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return {}
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1))))
        cv2.drawFrameAxes(image, matrix, distortion, rvec, tvec, marker_length_m * 0.5)
        return {
            "rvec": rvec.reshape(3).astype(float).tolist(),
            "tvec_m": tvec.reshape(3).astype(float).tolist(),
            "reprojection_error_px": error,
        }

    def _calibration_status(self) -> dict[str, object]:
        expected = {
            "serial_number": str(self.calibration["serial_number"]),
            "image_shape": [int(self.calibration["height"]), int(self.calibration["width"])],
            "fps": int(self.calibration["fps"]),
        }
        try:
            config = self.camera.teleimager_config()
            actual = config["head_camera"]
            actual_profile = {
                "serial_number": str(actual.get("serial_number")),
                "image_shape": list(actual.get("image_shape", [])),
                "fps": int(actual.get("fps", 0)),
            }
            valid = actual_profile == expected
            error = None if valid else "Teleimager head-camera profile does not match calibration"
        except (CameraUnavailable, KeyError, TypeError, ValueError, AttributeError) as error_value:
            actual_profile = None
            valid = False
            error = str(error_value)
        return {"valid": valid, "expected": expected, "actual": actual_profile, "error": error}

    @staticmethod
    def _empty_result(
        dictionary: str = "DICT_4X4_50",
        marker_length_mm: float | None = None,
        calibration_valid: bool = False,
    ) -> dict[str, object]:
        return {
            "observed_at": None,
            "frame_size": None,
            "camera_serial_number": None,
            "dictionary": dictionary,
            "marker_length_mm": marker_length_mm,
            "calibration_valid": calibration_valid,
            "coordinate_frame": "opencv_camera_x_right_y_down_z_forward",
            "markers": [],
        }
