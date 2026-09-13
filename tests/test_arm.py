import os
import unittest
from pathlib import Path

import numpy as np

from classic_control.arm import (
    HAND_FINGER_AXIS_LOCAL,
    HAND_PALM_NORMAL_LOCAL,
    NORMAL_ARM_Q,
    ArmPlanner,
    apply_nudge,
    axis_rotation,
    rotation_error_rad,
    rotation_from_rpy_degrees,
    slerp,
)


class ArmMathTest(unittest.TestCase):
    def test_translation_is_in_base_frame(self):
        transform = np.eye(4)
        transform[:3, :3] = axis_rotation(2, np.pi / 2)
        moved = apply_nudge(transform, "translation", 0, 0.01)
        np.testing.assert_allclose(moved[:3, 3], (0.01, 0, 0))
        np.testing.assert_allclose(moved[:3, :3], transform[:3, :3])

    def test_rotation_is_wrist_local(self):
        transform = np.eye(4)
        transform[:3, :3] = axis_rotation(2, np.pi / 2)
        moved = apply_nudge(transform, "rotation", 0, 0.2)
        np.testing.assert_allclose(moved[:3, :3], transform[:3, :3] @ axis_rotation(0, 0.2))
        np.testing.assert_allclose(moved[:3, :3].T @ moved[:3, :3], np.eye(3), atol=1e-12)

    def test_slerp_endpoints(self):
        start = np.eye(3)
        target = rotation_from_rpy_degrees(20, -10, 45)
        np.testing.assert_allclose(slerp(start, target, 0), start, atol=1e-12)
        self.assertLess(rotation_error_rad(slerp(start, target, 1), target), 1e-7)


@unittest.skipUnless(
    os.environ.get("XR_TELEOPERATE_ROOT"),
    "XR_TELEOPERATE_ROOT is required for Pinocchio FK",
)
class NormalPoseFkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = ArmPlanner(Path(os.environ["XR_TELEOPERATE_ROOT"]))

    def test_shoulders_elbows_and_wrist_roll_keep_baseline_values(self):
        np.testing.assert_allclose(
            NORMAL_ARM_Q[[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]],
            (0.0, 0.19, 0.0, 0.0, 0.0, 0.0, -0.19, 0.0, 0.0, 0.0),
        )

    def test_fingers_point_up_and_palms_face_forward(self):
        robot_up = np.array((0.0, 0.0, 1.0))
        robot_forward = np.array((1.0, 0.0, 0.0))
        tolerance_rad = np.deg2rad(0.25)

        for side, transform in zip(
            ("left", "right"), self.planner.wrist_poses(NORMAL_ARM_Q)
        ):
            rotation = transform[:3, :3]
            finger_axis = rotation @ HAND_FINGER_AXIS_LOCAL
            palm_normal = rotation @ HAND_PALM_NORMAL_LOCAL
            finger_error = np.arccos(np.clip(finger_axis @ robot_up, -1.0, 1.0))
            palm_error = np.arccos(
                np.clip(palm_normal @ robot_forward, -1.0, 1.0)
            )
            with self.subTest(side=side):
                self.assertLess(finger_error, tolerance_rad)
                self.assertLess(palm_error, tolerance_rad)


if __name__ == "__main__":
    unittest.main()
