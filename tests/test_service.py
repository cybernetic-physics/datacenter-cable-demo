import time
import unittest
from types import SimpleNamespace

import numpy as np

from classic_control.arm import NORMAL_ARM_Q, ArmWaypoint
from classic_control.aruco_targeting import ResolvedArucoTarget
from classic_control.hardware import RobotState
from classic_control.models import ArmSide, ElbowBias, MarkerOffset, PoseTarget
from classic_control.service import ControlService

OPEN = {
    "thumb_rotate": 0.0, "thumb_1": 0.0, "thumb_2": 0.0,
    "index_0": 0.0, "index_1": 0.0, "middle_0": 0.0, "middle_1": 0.0,
}


class FakeBackend:
    def __init__(self):
        self.q = np.zeros(35)
        self.q[15:29] = NORMAL_ARM_Q
        self.dq = np.zeros(35)
        self.hands = {"left": OPEN.copy(), "right": OPEN.copy()}
        self.acquired = False
        self.releases = 0
        self.body_targets = []
        self.hand_targets = []

    def connect(self): pass
    def acquire(self): self.acquired = True
    def state(self): return RobotState(self.q.copy(), self.dq.copy(), {k:v.copy() for k,v in self.hands.items()}, self.acquired)
    def set_body_target(self, arm_q, arm_tau, waist_q=None):
        self.q[15:29] = arm_q
        if waist_q is not None: self.q[12:15] = waist_q
        self.body_targets.append((arm_q.copy(), None if waist_q is None else waist_q.copy()))
    def set_hand_target(self, side, joints): self.hands[side] = joints.copy(); self.hand_targets.append((side, joints.copy()))
    def release(self): self.acquired = False; self.releases += 1
    def close(self): self.release()


class FakePlanner:
    def reset(self, q): self.reset_q = np.asarray(q).copy()
    def wrist_poses(self, q):
        left, right = np.eye(4), np.eye(4)
        left[0, 3], right[0, 3] = q[0], q[7]
        return left, right
    def plan(self, start_q, start_dq, start_left, start_right, end_left, end_right, duration_s, moving_side, elbow):
        q = np.asarray(start_q).copy()
        q[0 if moving_side.value == "left" else 7] += 0.01
        return SimpleNamespace(waypoints=(ArmWaypoint(q, np.zeros(14)),), duration_s=0.01,
                               left_target=end_left, right_target=end_right)


class FakeGrasps:
    def load(self): return {}


class FakeArucoTargeting:
    def __init__(self):
        self.config = SimpleNamespace(max_waist_error_rad=0.035)
        self.offset = None
        self.gate = None

    def resolve(self, marker_id, offset):
        self.offset = offset
        wrist = np.eye(4); wrist[:3, 3] = (0.25, -0.20, 0.35)
        return ResolvedArucoTarget(marker_id, 0.1, 0.2, np.eye(4), np.eye(4), np.eye(4), wrist)

    def resolve_gate(self, gate_index, offset):
        self.gate = gate_index
        self.offset = offset
        wrist = np.eye(4); wrist[:3, 3] = (0.15, -0.25, 0.45)
        return SimpleNamespace(gate_index=gate_index, base_from_wrist=wrist)

    @staticmethod
    def resolution_json(resolved):
        return {"marker_id": resolved.marker_id}

    @staticmethod
    def gate_resolution_json(resolved):
        return {"target_kind": "gate", "gate_index": resolved.gate_index}


def wait_idle(service, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if service.telemetry()["active_command"] is None: return
        time.sleep(0.005)
    raise AssertionError("motion did not finish")


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.service = ControlService(self.backend, FakePlanner(), FakeGrasps())
        self.service.start(); self.assertTrue(self.service.attach("owner"))

    def tearDown(self): self.service.close()

    def test_jog_acquires_control(self):
        self.service.command_jog("owner", "left", "translation", 0, 0.01, 0.1, "auto")
        wait_idle(self.service)
        self.assertTrue(self.backend.acquired)
        self.assertEqual(self.service.visualization_snapshot().planned_arm_q.shape, (14,))

    def test_stop_motion_holds_without_releasing_control(self):
        self.service.command_jog("owner", "left", "translation", 0, 0.01, 0.1, "auto")
        self.service.stop_motion("owner")
        self.assertTrue(self.backend.acquired)

    def test_ik_rejection_does_not_latch_control_fault(self):
        def reject(*_args):
            raise RuntimeError("IK target is unreachable")

        self.service.planner.plan = reject
        target = PoseTarget(ArmSide.RIGHT, (0.3, -0.2, 0.1), (0, 0, 0), 1.0, ElbowBias.AUTO)
        with self.assertRaisesRegex(RuntimeError, "unreachable"):
            self.service.command_pose("owner", target)

        self.assertIsNone(self.service.telemetry()["fault"])
        self.assertTrue(self.backend.acquired)

    def test_normal_sets_waist_and_arms(self):
        self.backend.q[12:15] = (0.001, -0.001, 0.001)
        self.service.command_normal("owner", 0.05)
        wait_idle(self.service, 2)
        np.testing.assert_allclose(self.backend.body_targets[-1][0], NORMAL_ARM_Q)
        np.testing.assert_allclose(self.backend.body_targets[-1][1], np.zeros(3))

    def test_disconnect_releases_robot(self):
        self.service.command_hand("owner", {"right": OPEN}, 0)
        wait_idle(self.service)
        self.service.detach("owner")
        self.assertFalse(self.backend.acquired)
        self.assertEqual(self.backend.releases, 1)

    def test_second_browser_is_rejected(self):
        self.assertFalse(self.service.attach("other"))

    def test_aruco_command_routes_through_absolute_pose(self):
        targeting = FakeArucoTargeting()
        self.service.aruco_targeting = targeting
        captured = []
        self.service.command_pose = lambda owner, target: captured.append((owner, target))
        offset = MarkerOffset((0, 0, 0.1), (0, 0, 0))

        self.service.command_aruco("owner", "right", 3, offset, 2.0, "down")

        self.assertEqual(captured[0][0], "owner")
        self.assertEqual(captured[0][1].xyz, (0.25, -0.2, 0.35))
        self.assertEqual(captured[0][1].side.value, "right")
        self.assertIs(targeting.offset, offset)
        self.assertEqual(self.service.telemetry()["aruco_target"]["marker_id"], 3)

    def test_aruco_command_requires_neutral_waist(self):
        self.service.aruco_targeting = FakeArucoTargeting()
        self.backend.q[12] = 0.04
        with self.assertRaisesRegex(RuntimeError, "neutral waist"):
            self.service.command_aruco("owner", "right", 3, None, 2.0, "auto")

    def test_gate_command_routes_through_absolute_pose(self):
        targeting = FakeArucoTargeting()
        self.service.aruco_targeting = targeting
        captured = []
        self.service.command_pose = lambda owner, target: captured.append((owner, target))

        self.service.command_gate("owner", "right", 23, None, 3.0, "auto")

        self.assertEqual(targeting.gate, 23)
        self.assertEqual(captured[0][1].xyz, (0.15, -0.25, 0.45))
        self.assertEqual(self.service.telemetry()["aruco_target"]["gate_index"], 23)


if __name__ == "__main__":
    unittest.main()
