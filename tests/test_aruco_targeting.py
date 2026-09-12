import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from classic_control.aruco_targeting import (
    ArucoTargeting,
    compose_target,
    pose_transform,
)
from classic_control.models import MarkerOffset


def targeting_payload() -> dict[str, object]:
    return {
        "version": 1,
        "aruco_targeting": {
            "camera_serial_number": "test-camera",
            "base_from_internal_camera": {
                "source": "test",
                "matrix": np.eye(4).tolist(),
            },
            "default_offset": {"xyz_m": [0, 0, 0.08], "rpy_deg": [0, 0, 0]},
            "max_detection_age_s": 0.5,
            "max_reprojection_error_px": 2.0,
            "max_waist_error_rad": 0.035,
        },
    }


class StubObservations:
    def __init__(self, result, marker, age_s):
        self.value = result, marker, age_s

    def marker_observation(self, marker_id):
        return self.value


class ArucoTransformTest(unittest.TestCase):
    def test_camera_to_base_composition(self):
        base_camera = pose_transform((1, 2, 3), (0, 0, 90))
        camera_marker = pose_transform((0.2, 0, 0), (0, 0, 0))

        base_marker, _ = compose_target(
            base_camera, camera_marker, MarkerOffset((0, 0, 0), (0, 0, 0))
        )

        np.testing.assert_allclose(base_marker[:3, 3], (1, 2.2, 3), atol=1e-12)
        np.testing.assert_allclose(base_marker[:3, :3], base_camera[:3, :3], atol=1e-12)

    def test_marker_frame_offset_rotates_with_marker(self):
        base_marker = pose_transform((0.4, -0.2, 0.8), (0, 90, 0))
        offset = MarkerOffset((0, 0, 0.08), (10, 20, 30))

        _, base_wrist = compose_target(np.eye(4), base_marker, offset)

        expected = base_marker @ pose_transform(offset.xyz_m, offset.rpy_deg)
        np.testing.assert_allclose(base_wrist, expected, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(base_wrist[:3, 3] - base_marker[:3, 3]), 0.08)


class ArucoValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary.name) / "targeting.yaml"
        self.config_path.write_text(yaml.safe_dump(targeting_payload()), encoding="utf-8")
        self.result = {
            "observed_at": "now",
            "calibration_valid": True,
            "camera_serial_number": "test-camera",
        }
        self.marker = {
            "id": 3,
            "rvec": [0, 0, 0],
            "tvec_m": [0.1, 0.2, 0.3],
            "reprojection_error_px": 0.5,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def resolver(self, result=None, marker="default", age_s=0.1):
        selected_marker = self.marker if marker == "default" else marker
        source = StubObservations(self.result if result is None else result, selected_marker, age_s)
        return ArucoTargeting(source, self.config_path)

    def test_resolves_valid_metric_detection(self):
        resolved = self.resolver().resolve(3)
        np.testing.assert_allclose(resolved.base_from_marker[:3, 3], (0.1, 0.2, 0.3))
        np.testing.assert_allclose(resolved.base_from_wrist[:3, 3], (0.1, 0.2, 0.38))

    def test_rejects_missing_stale_invalid_and_high_error_detection(self):
        cases = (
            (self.resolver(marker=None), "not detected"),
            (self.resolver(age_s=0.6), "stale"),
            (
                self.resolver(result={"observed_at": "now", "calibration_valid": False}),
                "calibration",
            ),
            (
                self.resolver(marker={**self.marker, "reprojection_error_px": 2.1}),
                "too high",
            ),
            (
                self.resolver(result={**self.result, "camera_serial_number": "other"}),
                "camera identity",
            ),
            (self.resolver(marker={**self.marker, "tvec_m": None}), "metric marker pose"),
        )
        for resolver, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                resolver.resolve(3)


if __name__ == "__main__":
    unittest.main()
