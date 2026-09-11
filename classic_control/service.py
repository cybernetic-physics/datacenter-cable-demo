"""Thread-safe orchestration between browser commands, planning, and hardware."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

import numpy as np

from .arm import (
    NORMAL_ARM_Q,
    NORMAL_WAIST_Q,
    ArmPlanner,
    apply_nudge,
    joint_trajectory,
    rotation_from_rpy_degrees,
    rpy_degrees_from_rotation,
)
from .grasps import JOINT_NAMES, GraspStore, validate_hand
from .hardware import RobotBackend
from .models import ArmSide, ElbowBias, PoseTarget

DEADMAN_TIMEOUT_S = 0.35
NORMAL_SPEED_RAD_S = 0.08


class ControlService:
    def __init__(self, backend: RobotBackend, planner: ArmPlanner, grasps: GraspStore):
        self.backend = backend
        self.planner = planner
        self.grasps = grasps
        self._lock = threading.RLock()
        self._planner_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._owner: str | None = None
        self._deadman = False
        self._heartbeat = 0.0
        self._motion_cancel = threading.Event()
        self._motion_thread: threading.Thread | None = None
        self._active_command: str | None = None
        self._fault: str | None = None
        self._left_target: np.ndarray | None = None
        self._right_target: np.ndarray | None = None
        self._hand_targets: dict[str, dict[str, float] | None] = {"left": None, "right": None}

    def start(self) -> None:
        self.backend.connect()

    def close(self) -> None:
        self._cancel_motion(wait=True)
        self.backend.close()

    def attach(self, owner: str) -> bool:
        with self._lock:
            if self._owner not in (None, owner):
                return False
            self._owner = owner
            self._heartbeat = time.monotonic()
            return True

    def detach(self, owner: str) -> None:
        with self._lock:
            if self._owner != owner:
                return
        try:
            self.release_control(owner)
        finally:
            with self._lock:
                if self._owner == owner:
                    self._owner = None

    def set_deadman(self, owner: str, active: bool) -> None:
        self._require_owner(owner)
        with self._lock:
            self._heartbeat = time.monotonic()
            self._deadman = bool(active)
        if not active:
            self._cancel_motion(wait=False)

    def watchdog(self) -> None:
        with self._lock:
            expired = self._deadman and time.monotonic() - self._heartbeat > DEADMAN_TIMEOUT_S
            if expired:
                self._deadman = False
        if expired:
            self._cancel_motion(wait=False)

    def _require_owner(self, owner: str) -> None:
        with self._lock:
            if self._owner != owner:
                raise RuntimeError("this browser does not own the control session")

    def _require_motion_authority(self, owner: str) -> None:
        self._require_owner(owner)
        with self._lock:
            if not self._deadman:
                raise RuntimeError("hold the deadman control before requesting motion")
            if self._fault is not None:
                raise RuntimeError(f"control is faulted: {self._fault}; release control before retrying")

    def _ensure_acquired(self, owner: str) -> None:
        self._require_motion_authority(owner)
        try:
            state = self.backend.state()
        except Exception as error:
            self._set_fault(error)
            raise
        if state.fault:
            with self._lock:
                self._fault = state.fault
            raise RuntimeError(state.fault)
        if not state.acquired:
            try:
                self.backend.acquire()
                state = self.backend.state()
            except Exception as error:
                self._set_fault(error)
                raise
            arm_q = state.body_q[15:29]
            with self._planner_lock:
                self.planner.reset(arm_q)
                left, right = self.planner.wrist_poses(arm_q)
            with self._lock:
                self._left_target, self._right_target = left, right
                self._hand_targets = {
                    side: None if state.hands[side] is None else state.hands[side].copy()
                    for side in ("left", "right")
                }
                self._fault = None

    def _set_fault(self, error: Exception) -> None:
        self._cancel_motion(wait=False)
        with self._lock:
            self._fault = str(error)

    def _cancel_motion(self, wait: bool) -> None:
        self._motion_cancel.set()
        thread = self._motion_thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _install_motion(self, owner: str, label: str, runner) -> None:
        self._require_motion_authority(owner)
        self._cancel_motion(wait=True)
        cancel = threading.Event()
        with self._lock:
            self._motion_cancel = cancel
            self._active_command = label

        def execute() -> None:
            try:
                runner(cancel)
            except Exception as error:
                with self._lock:
                    self._fault = str(error)
            finally:
                with self._lock:
                    if self._motion_cancel is cancel:
                        self._active_command = None

        thread = threading.Thread(target=execute, name=f"motion-{label}", daemon=True)
        self._motion_thread = thread
        thread.start()

    def _authority_still_valid(self, owner: str) -> bool:
        with self._lock:
            return self._owner == owner and self._deadman and self._fault is None

    def command_jog(self, owner: str, side: str, mode: str, axis: int,
                    delta: float, duration_s: float, elbow: str) -> None:
        moving_side = ArmSide(side)
        bias = ElbowBias(elbow)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        if not np.isfinite(delta) or abs(delta) > (0.15 if mode == "translation" else np.deg2rad(45)):
            raise ValueError("jog delta is invalid or too large")
        with self._command_lock:
            self._ensure_acquired(owner)
            self._cancel_motion(wait=True)
            state = self.backend.state()
            with self._planner_lock:
                self.planner.reset(state.body_q[15:29])
                start_left, start_right = self.planner.wrist_poses(state.body_q[15:29])
            end_left, end_right = start_left.copy(), start_right.copy()
            selected = end_left if moving_side == ArmSide.LEFT else end_right
            selected[:] = apply_nudge(selected, mode, axis, delta)
            try:
                with self._planner_lock:
                    plan = self.planner.plan(state.body_q[15:29], state.body_dq[15:29],
                                             start_left, start_right, end_left, end_right,
                                             duration_s, moving_side, bias)
            except Exception as error:
                self._set_fault(error)
                raise
            if not self._authority_still_valid(owner):
                return
            self._run_arm_plan(owner, f"{side}-{mode}-jog", plan)

    def command_pose(self, owner: str, target: PoseTarget) -> None:
        target.validate()
        with self._command_lock:
            self._ensure_acquired(owner)
            self._cancel_motion(wait=True)
            state = self.backend.state()
            with self._planner_lock:
                self.planner.reset(state.body_q[15:29])
                start_left, start_right = self.planner.wrist_poses(state.body_q[15:29])
            end_left, end_right = start_left.copy(), start_right.copy()
            selected = end_left if target.side == ArmSide.LEFT else end_right
            selected[:3, 3] = target.xyz
            selected[:3, :3] = rotation_from_rpy_degrees(*target.rpy_deg)
            try:
                with self._planner_lock:
                    plan = self.planner.plan(state.body_q[15:29], state.body_dq[15:29],
                                             start_left, start_right, end_left, end_right,
                                             target.duration_s, target.side, target.elbow)
            except Exception as error:
                self._set_fault(error)
                raise
            if not self._authority_still_valid(owner):
                return
            self._run_arm_plan(owner, f"{target.side.value}-absolute", plan)

    def _run_arm_plan(self, owner: str, label: str, plan) -> None:
        def runner(cancel: threading.Event) -> None:
            period = plan.duration_s / len(plan.waypoints)
            deadline = time.monotonic()
            for waypoint in plan.waypoints:
                if cancel.is_set() or not self._authority_still_valid(owner):
                    return
                self.backend.set_body_target(waypoint.q, waypoint.tau)
                with self._planner_lock:
                    left_target, right_target = self.planner.wrist_poses(waypoint.q)
                with self._lock:
                    self._left_target, self._right_target = left_target, right_target
                deadline += period
                cancel.wait(max(0.0, deadline - time.monotonic()))
            with self._lock:
                self._left_target = plan.left_target.copy()
                self._right_target = plan.right_target.copy()

        self._install_motion(owner, label, runner)

    def command_normal(self, owner: str, requested_duration_s: float = 20.0) -> None:
        if not np.isfinite(requested_duration_s) or not 0.05 <= requested_duration_s <= 60.0:
            raise ValueError("normal duration must be in [0.05, 60] seconds")
        with self._command_lock:
            self._ensure_acquired(owner)
            self._cancel_motion(wait=True)
            state = self.backend.state()
            current_arm = state.body_q[15:29]
            current_waist = state.body_q[12:15]
            largest = float(np.max(np.abs(np.concatenate((NORMAL_ARM_Q - current_arm,
                                                          NORMAL_WAIST_Q - current_waist)))))
            duration = max(float(requested_duration_s), largest / NORMAL_SPEED_RAD_S)
            trajectory = joint_trajectory(current_arm, NORMAL_ARM_Q, duration)
            with self._planner_lock:
                normal_left, normal_right = self.planner.wrist_poses(NORMAL_ARM_Q)

            def runner(cancel: threading.Event) -> None:
                period = duration / len(trajectory)
                deadline = time.monotonic()
                for index, waypoint in enumerate(trajectory, 1):
                    if cancel.is_set() or not self._authority_still_valid(owner):
                        return
                    alpha = index / len(trajectory)
                    waist = (1.0 - alpha) * current_waist + alpha * NORMAL_WAIST_Q
                    self.backend.set_body_target(waypoint.q, waypoint.tau, waist)
                    with self._planner_lock:
                        left_target, right_target = self.planner.wrist_poses(waypoint.q)
                    with self._lock:
                        self._left_target, self._right_target = left_target, right_target
                    deadline += period
                    cancel.wait(max(0.0, deadline - time.monotonic()))
                with self._planner_lock:
                    self.planner.reset(NORMAL_ARM_Q)
                with self._lock:
                    self._left_target, self._right_target = normal_left, normal_right

            self._install_motion(owner, "normal-pose", runner)

    def command_hand(self, owner: str, targets: Mapping[str, Mapping[str, float]], duration_s: float) -> None:
        if not np.isfinite(duration_s) or not 0.0 <= duration_s <= 10.0:
            raise ValueError("hand duration must be in [0, 10] seconds")
        validated = {side: validate_hand(side, values) for side, values in targets.items()}
        if not validated:
            raise ValueError("at least one hand target is required")
        with self._command_lock:
            self._ensure_acquired(owner)
            self._cancel_motion(wait=True)
            state = self.backend.state()
            starts: dict[str, dict[str, float]] = {}
            for side in validated:
                with self._lock:
                    existing = self._hand_targets[side]
                start = existing or state.hands[side]
                if start is None:
                    raise RuntimeError(f"no finite {side} Dex3 state received")
                starts[side] = start.copy()

            def runner(cancel: threading.Event) -> None:
                steps = max(1, round(max(duration_s, 0.025) * 40.0))
                deadline = time.monotonic()
                for index in range(1, steps + 1):
                    if cancel.is_set() or not self._authority_still_valid(owner):
                        return
                    alpha = index / steps
                    for side, goal in validated.items():
                        values = {name: (1.0 - alpha) * starts[side][name] + alpha * goal[name]
                                  for name in JOINT_NAMES}
                        self.backend.set_hand_target(side, values)
                        with self._lock:
                            self._hand_targets[side] = values.copy()
                    deadline += duration_s / steps if duration_s else 0.0
                    cancel.wait(max(0.0, deadline - time.monotonic()))
                with self._lock:
                    for side, goal in validated.items():
                        self._hand_targets[side] = goal.copy()

            self._install_motion(owner, "dex3-target", runner)

    def command_grasp(self, owner: str, name: str, sides: list[str], duration_s: float) -> None:
        grasp = self.grasps.load().get(name)
        if grasp is None:
            raise KeyError(name)
        selected = sides or list(grasp.hands)
        targets = {side: grasp.hands[side] for side in selected if side in grasp.hands}
        if not targets:
            raise ValueError(f"grasp {name!r} has no targets for the selected hands")
        self.command_hand(owner, targets, duration_s)

    def release_control(self, owner: str) -> None:
        self._require_owner(owner)
        with self._lock:
            self._deadman = False
        self._cancel_motion(wait=True)
        try:
            self.backend.release()
        finally:
            with self._lock:
                self._active_command = None
                self._fault = None
                self._left_target = self._right_target = None
                self._hand_targets = {"left": None, "right": None}

    @staticmethod
    def _pose_json(transform: np.ndarray) -> dict[str, list[float]]:
        return {
            "xyz": np.round(transform[:3, 3], 6).tolist(),
            "rpy_deg": np.round(rpy_degrees_from_rotation(transform[:3, :3]), 4).tolist(),
        }

    def telemetry(self) -> dict[str, object]:
        self.watchdog()
        try:
            state = self.backend.state()
            arm_q = state.body_q[15:29]
            with self._planner_lock:
                measured_left, measured_right = self.planner.wrist_poses(arm_q)
            hardware_fault = state.fault
            state_error = None
        except Exception as error:
            state = None
            measured_left = measured_right = None
            hardware_fault = None
            state_error = str(error)
        if state is not None and state.acquired and hardware_fault:
            self._set_fault(RuntimeError(hardware_fault))
        with self._lock:
            fault = self._fault or hardware_fault or state_error
            return {
                "connected": state is not None,
                "owner": self._owner is not None,
                "deadman": self._deadman,
                "acquired": bool(state and state.acquired),
                "active_command": self._active_command,
                "fault": fault,
                "arms": None if state is None else {
                    "left": {
                        "measured": self._pose_json(measured_left),
                        "target": None if self._left_target is None else self._pose_json(self._left_target),
                    },
                    "right": {
                        "measured": self._pose_json(measured_right),
                        "target": None if self._right_target is None else self._pose_json(self._right_target),
                    },
                    "waist_q": np.round(state.body_q[12:15], 5).tolist(),
                },
                "hands": None if state is None else state.hands,
            }
