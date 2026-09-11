import unittest

import numpy as np

from classic_control.arm import (
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


if __name__ == "__main__":
    unittest.main()
