import unittest

import numpy as np

from classic_control.models import RobotVisualizationState
from classic_control.simulation import (
    BODY_JOINT_NAMES,
    HAND_JOINT_NAMES,
    MuJoCoDebugView,
    joint_value_map,
    world_from_base,
)


class EmptyTargeting:
    def visualized_markers(self):
        return ()


class EmptyProvider:
    def visualization_snapshot(self):
        raise RuntimeError("not used")


class SimulationMathTest(unittest.TestCase):
    def test_camera_view_orbits_zooms_clamps_and_resets(self):
        view = MuJoCoDebugView(EmptyProvider(), EmptyTargeting(), None)

        changed = view.update_view(30.0, 200.0, 2.0)

        self.assertEqual(changed["azimuth_deg"], 165.0)
        self.assertEqual(changed["elevation_deg"], 89.0)
        self.assertEqual(changed["distance_m"], 2.9)
        reset = view.update_view(reset=True)
        self.assertEqual(reset["azimuth_deg"], 135.0)
        self.assertEqual(reset["elevation_deg"], -20.0)
        self.assertEqual(reset["distance_m"], 1.45)

    def test_maps_body_and_dex3_joints_by_name(self):
        body = np.arange(35, dtype=float) / 10
        hands = {
            "left": {
                "thumb_rotate": 0.1, "thumb_1": 0.2, "thumb_2": 0.3,
                "index_0": 0.4, "index_1": 0.5, "middle_0": 0.6, "middle_1": 0.7,
            },
            "right": None,
        }
        state = RobotVisualizationState(body, hands, None, None)

        values = joint_value_map(state)

        self.assertEqual(values["waist_yaw_joint"], body[12])
        self.assertEqual(values["right_wrist_yaw_joint"], body[28])
        self.assertEqual(values[HAND_JOINT_NAMES["left"]["index_0"]], 0.4)
        self.assertEqual(len([name for name in BODY_JOINT_NAMES if name in values]), 29)

    def test_composes_pelvis_relative_pose_into_world(self):
        world_base = np.eye(4)
        world_base[:3, 3] = (0, 0, 0.793)
        base_marker = np.eye(4)
        base_marker[:3, 3] = (0.4, -0.2, 0.1)

        result = world_from_base(world_base, base_marker)

        np.testing.assert_allclose(result[:3, 3], (0.4, -0.2, 0.893))

    def test_missing_model_fails_without_touching_state_provider(self):
        view = MuJoCoDebugView(EmptyProvider(), EmptyTargeting(), None)
        stream = view.stream()
        try:
            with self.assertRaises(StopIteration):
                next(stream)
            self.assertEqual(view.status()["state"], "error")
        finally:
            stream.close()
            view.close()


if __name__ == "__main__":
    unittest.main()
