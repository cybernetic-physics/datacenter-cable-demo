import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from classic_control.aruco import ArucoVision

CALIBRATION = {
    "version": 1,
    "cameras": {
        "internal": {
            "serial_number": "test-camera",
            "width": 640,
            "height": 480,
            "fps": 30,
            "camera_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    },
}


def marker_jpeg(marker_id: int = 7) -> bytes:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    image[160:320, 240:400] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    return jpeg.tobytes()


class StubCamera:
    def __init__(self, profile_matches: bool = True, frames: list[bytes] | None = None):
        self.profile_matches = profile_matches
        self.frames = frames or [marker_jpeg()]

    def jpeg_stream(self, source_id):
        if source_id != "internal":
            raise KeyError(source_id)
        return iter(self.frames)

    def teleimager_config(self):
        return {"head_camera": {
            "serial_number": "test-camera" if self.profile_matches else "other-camera",
            "image_shape": [480, 640],
            "fps": 30,
        }}


class ArucoVisionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.calibration_path = Path(self.temporary.name) / "calibration.yaml"
        self.calibration_path.write_text(yaml.safe_dump(CALIBRATION), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_detects_marker_and_estimates_pose(self):
        vision = ArucoVision(StubCamera(), self.calibration_path)
        vision.configure("DICT_4X4_50", 50.0)
        chunks = list(vision.annotated_stream())
        result = vision.latest()

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].startswith(b"--frame\r\nContent-Type: image/jpeg"))
        self.assertEqual([marker["id"] for marker in result["markers"]], [7])
        self.assertEqual(len(result["markers"][0]["corners_px"]), 4)
        self.assertGreater(result["markers"][0]["tvec_m"][2], 0)
        self.assertAlmostEqual(result["markers"][0]["tvec_m"][2], 0.188, delta=0.01)
        self.assertTrue(result["calibration_valid"])

    def test_profile_mismatch_disables_metric_pose(self):
        vision = ArucoVision(StubCamera(profile_matches=False), self.calibration_path)
        vision.configure("DICT_4X4_50", 50.0)
        list(vision.annotated_stream())
        marker = vision.latest()["markers"][0]

        self.assertIsNone(marker["tvec_m"])
        self.assertFalse(vision.latest()["calibration_valid"])

    def test_no_marker_and_invalid_jpeg_are_safe(self):
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", blank)
        self.assertTrue(ok)
        vision = ArucoVision(
            StubCamera(frames=[b"not a jpeg", jpeg.tobytes()]), self.calibration_path
        )

        chunks = list(vision.annotated_stream())

        self.assertEqual(len(chunks), 1)
        self.assertEqual(vision.latest()["markers"], [])

    def test_rejects_invalid_configuration(self):
        vision = ArucoVision(StubCamera(), self.calibration_path)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            vision.configure("NOT_A_DICTIONARY", None)
        with self.assertRaisesRegex(ValueError, "marker_length_mm"):
            vision.configure("DICT_4X4_50", -1)


if __name__ == "__main__":
    unittest.main()
