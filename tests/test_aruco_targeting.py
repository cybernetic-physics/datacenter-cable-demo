import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from classic_control.aruco_targeting import (
    ArucoTargeting,
    EthernetGateGrid,
    compose_target,
    pose_transform,
)
from classic_control.models import MarkerOffset


def targeting_payload() -> dict[str, object]:
    return {
        "version": 1,
        "aruco_targeting": {
            "camera_serial_number": "test-camera",
            "right_rack_marker_id": 3,
            "base_from_internal_camera": {
                "source": "test",
                "matrix": np.eye(4).tolist(),
            },
            "default_offset": {"xyz_m": [0, 0, 0.08], "rpy_deg": [0, 90, 90]},
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

    def latest(self):
        return self.value[0]


class ArucoTransformTest(unittest.TestCase):
    def test_gate_zero_uses_marker_edges_and_gate_center(self):
        pose = EthernetGateGrid(0.05).marker_from_gate(0)

        np.testing.assert_allclose(pose[:3, 3], (0.0105, 0.0525, 0.0))
        np.testing.assert_allclose(pose[:3, :3], np.eye(3))

    def test_gate_pitch_and_gate_23(self):
        grid = EthernetGateGrid(0.05)
        gate_0 = grid.marker_from_gate(0)
        gate_1 = grid.marker_from_gate(1)
        gate_23 = grid.marker_from_gate(23)

        self.assertAlmostEqual(gate_0[0, 3] - gate_1[0, 3], 0.019)
        self.assertAlmostEqual(gate_23[0, 3], 0.0105 - 23 * 0.019)
        self.assertAlmostEqual(gate_23[1, 3], 0.0525)

    def test_gate_grid_uses_detected_marker_size(self):
        small = EthernetGateGrid(0.05).marker_from_gate(0)
        large = EthernetGateGrid(0.10).marker_from_gate(0)

        np.testing.assert_allclose(
            large[:3, 3] - small[:3, 3], (-0.025, 0.025, 0.0)
        )

    def test_gate_grid_rejects_invalid_indices(self):
        grid = EthernetGateGrid(0.05)
        for value in (-1, 24, 1.5, True):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                grid.marker_from_gate(value)

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

    def test_default_tool_orientation_points_wrist_x_into_marker(self):
        marker_wrist = pose_transform((0, 0, 0.08), (0, 90, 90))
        np.testing.assert_allclose(marker_wrist[:3, 0], (0, 0, -1), atol=1e-12)
        np.testing.assert_allclose(marker_wrist[:3, 2], (0, 1, 0), atol=1e-12)


class ArucoValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary.name) / "targeting.yaml"
        self.config_path.write_text(yaml.safe_dump(targeting_payload()), encoding="utf-8")
        self.result = {
            "observed_at": "now",
            "calibration_valid": True,
            "camera_serial_number": "test-camera",
            "marker_length_mm": 50.0,
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

    def test_resolve_reuses_first_valid_pose_for_the_session(self):
        resolver = self.resolver()
        first = resolver.resolve(3)
        resolver.observations.value = (
            {**self.result, "observed_at": None},
            {**self.marker, "tvec_m": [9, 9, 9]},
            None,
        )

        second = resolver.resolve(3)

        np.testing.assert_allclose(second.base_from_marker, first.base_from_marker)

    def test_gate_resolution_reuses_saved_rack_marker_and_default_offset(self):
        resolver = self.resolver()

        resolved = resolver.resolve_gate(23)

        np.testing.assert_allclose(
            resolved.base_from_gate[:3, 3],
            (0.1 + 0.0105 - 23 * 0.019, 0.2 + 0.0525, 0.3),
        )
        np.testing.assert_allclose(
            resolved.base_from_wrist,
            resolved.base_from_gate
            @ pose_transform((0, 0, 0.08), (0, 90, 90)),
        )
        self.assertEqual(len(resolver.visualized_gates()), 24)

    def test_projects_saved_gate_footprints_into_rgb(self):
        resolver = self.resolver()
        resolver.resolve_gate(0)
        matrix = np.asarray(((600, 0, 320), (0, 600, 240), (0, 0, 1)), dtype=float)

        gates = resolver.projected_gates(matrix, np.zeros(5))

        self.assertEqual(len(gates), 24)
        np.testing.assert_allclose(gates[0].center_px, (541.0, 745.0), atol=1e-6)
        np.testing.assert_allclose(
            gates[0].corners_px,
            ((526, 760), (556, 760), (556, 730), (526, 730)),
            atol=1e-6,
        )

    def test_rgb_projection_omits_gates_behind_camera(self):
        marker = {
            **self.marker,
            "rvec": [-np.pi / 2, 0, 0],
            "tvec_m": [0, 0, 0.001],
        }
        resolver = self.resolver(marker=marker)
        resolver.resolve_gate(0)

        gates = resolver.projected_gates(np.eye(3), np.zeros(5))

        self.assertEqual(gates, ())

    def test_rgb_projection_rejects_invalid_calibration(self):
        resolver = self.resolver()
        resolver.resolve_gate(0)

        with self.assertRaisesRegex(ValueError, "calibration"):
            resolver.projected_gates(np.full((3, 3), np.nan), np.zeros(5))

    def test_visualization_latches_first_fresh_usable_marker_pose(self):
        result = {
            **self.result,
            "age_s": 0.1,
            "marker_length_mm": 50.0,
            "markers": [self.marker, {**self.marker, "id": 4, "reprojection_error_px": 2.1}],
        }
        resolver = self.resolver(result=result)

        markers = resolver.visualized_markers()

        self.assertEqual([marker.marker_id for marker in markers], [3])
        self.assertEqual(markers[0].size_m, 0.05)
        np.testing.assert_allclose(markers[0].base_from_marker[:3, 3], (0.1, 0.2, 0.3))
        resolver.observations.value[0]["age_s"] = 0.6
        resolver.observations.value[0]["markers"][0] = {
            **self.marker,
            "tvec_m": [0.8, 0.9, 1.0],
        }
        saved = resolver.visualized_markers()
        self.assertEqual([marker.marker_id for marker in saved], [3])
        np.testing.assert_allclose(saved[0].base_from_marker[:3, 3], (0.1, 0.2, 0.3))

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
