"""G1 dual-arm kinematics and checked trajectory planning."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import ArmSide, ElbowBias

WAYPOINT_HZ = 50.0
MAX_JOINT_STEP_RAD = 0.20
MAX_POSITION_ERROR_M = 0.02
MAX_ORIENTATION_ERROR_RAD = np.deg2rad(10.0)
FILTER_SETTLE_WAYPOINTS = 4
ELBOW_OFFSET_RAD = 0.08
ELBOW_FINITE_DIFFERENCE_RAD = 0.01
NORMAL_ARM_Q = np.array(
    [0.0, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, -0.19, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)
NORMAL_WAIST_Q = np.zeros(3, dtype=float)


@dataclass(frozen=True)
class ArmWaypoint:
    q: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class PlannedTrajectory:
    waypoints: tuple[ArmWaypoint, ...]
    duration_s: float
    left_target: np.ndarray
    right_target: np.ndarray
    maximum_joint_step: float
    maximum_position_error: float
    maximum_orientation_error: float


def axis_rotation(axis: int, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)), dtype=float)
    if axis == 1:
        return np.array(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)), dtype=float)
    if axis == 2:
        return np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)), dtype=float)
    raise ValueError("rotation axis must be 0, 1, or 2")


def apply_nudge(transform: np.ndarray, mode: str, axis: int, delta: float) -> np.ndarray:
    candidate = np.asarray(transform, dtype=float).copy()
    if candidate.shape != (4, 4) or not np.all(np.isfinite(candidate)):
        raise ValueError("wrist transform must be a finite 4x4 matrix")
    if not np.isfinite(delta):
        raise ValueError("nudge delta must be finite")
    if mode == "translation":
        candidate[axis, 3] += delta
    elif mode == "rotation":
        candidate[:3, :3] = candidate[:3, :3] @ axis_rotation(axis, delta)
    else:
        raise ValueError("mode must be translation or rotation")
    return candidate


def rotation_from_rpy_degrees(roll: float, pitch: float, yaw: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad((roll, pitch, yaw))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
         (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
         (-sp, cp * sr, cp * cr)),
        dtype=float,
    )


def rpy_degrees_from_rotation(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return np.rad2deg((roll, pitch, yaw))


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(((rotation[2, 1] - rotation[1, 2]) / scale,
                               (rotation[0, 2] - rotation[2, 0]) / scale,
                               (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale))
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array((0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                                   (rotation[0, 2] + rotation[2, 0]) / scale,
                                   (rotation[2, 1] - rotation[1, 2]) / scale))
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                                   (rotation[1, 2] + rotation[2, 1]) / scale,
                                   (rotation[0, 2] - rotation[2, 0]) / scale))
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(((rotation[0, 2] + rotation[2, 0]) / scale,
                                   (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                                   (rotation[1, 0] - rotation[0, 1]) / scale))
    return quaternion / np.linalg.norm(quaternion)


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
         (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
         (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        dtype=float,
    )


def slerp(start_rotation: np.ndarray, target_rotation: np.ndarray, alpha: float) -> np.ndarray:
    start = _quaternion_from_rotation(start_rotation)
    target = _quaternion_from_rotation(target_rotation)
    dot = float(np.dot(start, target))
    if dot < 0.0:
        target, dot = -target, -dot
    if dot > 0.9995:
        return _rotation_from_quaternion(start + alpha * (target - start))
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    result = (np.sin((1.0 - alpha) * angle) * start + np.sin(alpha * angle) * target) / np.sin(angle)
    return _rotation_from_quaternion(result)


def rotation_error_rad(actual: np.ndarray, target: np.ndarray) -> float:
    cosine = (np.trace(target.T @ actual) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def waypoint_count(duration_s: float) -> int:
    if not np.isfinite(duration_s) or not 0.05 <= duration_s <= 60.0:
        raise ValueError("duration must be finite and in [0.05, 60] seconds")
    return max(2, round(duration_s * WAYPOINT_HZ))


def joint_trajectory(start: Sequence[float], target: Sequence[float], duration_s: float) -> tuple[ArmWaypoint, ...]:
    start_q = np.asarray(start, dtype=float)
    target_q = np.asarray(target, dtype=float)
    if start_q.shape != (14,) or target_q.shape != (14,) or not np.all(np.isfinite((start_q, target_q))):
        raise ValueError("arm endpoints must be finite 14-element vectors")
    count = waypoint_count(duration_s)
    return tuple(
        ArmWaypoint((1.0 - index / count) * start_q + (index / count) * target_q, np.zeros(14))
        for index in range(1, count + 1)
    )


class ArmPlanner:
    """Stateful wrapper around Unitree's dual-arm IK and smoothing filter."""

    def __init__(self, xr_root: Path):
        sys.path.insert(0, str(xr_root))
        previous = Path.cwd()
        try:
            os.chdir(xr_root / "teleop")
            import pinocchio as pin
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

            self.pin = pin
            self.ik = G1_29_ArmIK()
        finally:
            os.chdir(previous)
        model = self.ik.reduced_robot.model
        if np.any(NORMAL_ARM_Q < model.lowerPositionLimit) or np.any(NORMAL_ARM_Q > model.upperPositionLimit):
            raise RuntimeError("neutral arm target violates an IK joint limit")

    def wrist_poses(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(q, dtype=float)
        model, data = self.ik.reduced_robot.model, self.ik.reduced_robot.data
        self.pin.forwardKinematics(model, data, values)
        self.pin.updateFramePlacements(model, data)
        return (data.oMf[model.getFrameId("L_ee")].homogeneous.copy(),
                data.oMf[model.getFrameId("R_ee")].homogeneous.copy())

    def reset(self, q: Sequence[float]) -> None:
        values = np.asarray(q, dtype=float).copy()
        self.ik.init_data = values.copy()
        self.ik.smooth_filter._data_queue = [values.copy()]
        self.ik.smooth_filter._filtered_data = values.copy()

    def _snapshot(self):
        return (np.asarray(self.ik.init_data).copy(),
                [np.asarray(item).copy() for item in self.ik.smooth_filter._data_queue],
                np.asarray(self.ik.smooth_filter._filtered_data).copy())

    def _restore(self, snapshot) -> None:
        initial, queue, filtered = snapshot
        self.ik.init_data = initial.copy()
        self.ik.smooth_filter._data_queue = [item.copy() for item in queue]
        self.ik.smooth_filter._filtered_data = filtered.copy()

    def _elbow_position(self, q: np.ndarray, side: ArmSide) -> np.ndarray:
        model, data = self.ik.reduced_robot.model, self.ik.reduced_robot.data
        self.pin.forwardKinematics(model, data, q)
        self.pin.updateFramePlacements(model, data)
        return data.oMf[model.getFrameId(f"{side.value}_elbow_link")].translation.copy()

    def _posture_seed(self, q: np.ndarray, side: ArmSide, bias: ElbowBias) -> np.ndarray:
        if bias == ElbowBias.AUTO:
            return q.copy()
        desired = {
            ElbowBias.DOWN: np.array((0.0, 0.0, -1.0)),
            ElbowBias.UP: np.array((0.0, 0.0, 1.0)),
            ElbowBias.OUT: np.array((0.0, 1.0 if side == ArmSide.LEFT else -1.0, 0.0)),
            ElbowBias.IN: np.array((0.0, -1.0 if side == ArmSide.LEFT else 1.0, 0.0)),
        }[bias]
        start = 0 if side == ArmSide.LEFT else 7
        baseline = self._elbow_position(q, side)
        best_index, best_projection = None, 0.0
        for index in range(start, start + 4):
            probe = q.copy()
            probe[index] += ELBOW_FINITE_DIFFERENCE_RAD
            projection = float(np.dot(self._elbow_position(probe, side) - baseline, desired))
            if abs(projection) > abs(best_projection):
                best_index, best_projection = index, projection
        if best_index is None or abs(best_projection) < 1e-8:
            return q.copy()
        seeded = q.copy()
        seeded[best_index] += np.copysign(ELBOW_OFFSET_RAD, best_projection)
        model = self.ik.reduced_robot.model
        return np.clip(seeded, model.lowerPositionLimit, model.upperPositionLimit)

    def _checked_ik(self, left: np.ndarray, right: np.ndarray, seed: np.ndarray,
                    previous: np.ndarray, dq: np.ndarray, index: int):
        solution_q, solution_tau = self.ik.solve_ik(left, right, seed, dq)
        try:
            stats = self.ik.opti.stats()
        except Exception as error:
            raise RuntimeError(f"could not read IK status at waypoint {index}") from error
        if not stats.get("success", False):
            raise RuntimeError(f"IK failed at waypoint {index}: {stats.get('return_status', 'unknown')}")
        solution_q = np.asarray(solution_q, dtype=float).reshape(-1)
        solution_tau = np.asarray(solution_tau, dtype=float).reshape(-1)
        if solution_q.shape != (14,) or solution_tau.shape != (14,) or not np.all(np.isfinite((solution_q, solution_tau))):
            raise RuntimeError(f"IK returned invalid values at waypoint {index}")
        model = self.ik.reduced_robot.model
        if np.any(solution_q < model.lowerPositionLimit - 1e-6) or np.any(solution_q > model.upperPositionLimit + 1e-6):
            raise RuntimeError(f"IK exceeded a joint limit at waypoint {index}")
        jump = float(np.max(np.abs(solution_q - previous)))
        if jump > MAX_JOINT_STEP_RAD:
            raise RuntimeError(f"IK joint jump {jump:.3f} exceeds {MAX_JOINT_STEP_RAD:.3f} rad")
        try:
            raw = np.asarray(self.ik.opti.value(self.ik.var_q), dtype=float).reshape(-1)
        except Exception:
            raw = solution_q
        if raw.shape != (14,) or not np.all(np.isfinite(raw)):
            raise RuntimeError(f"IK has no finite raw solution at waypoint {index}")
        return solution_q, solution_tau, raw, jump

    def plan(self, start_q: Sequence[float], start_dq: Sequence[float],
             start_left: np.ndarray, start_right: np.ndarray,
             end_left: np.ndarray, end_right: np.ndarray, duration_s: float,
             moving_side: ArmSide, elbow: ElbowBias) -> PlannedTrajectory:
        count = waypoint_count(duration_s)
        seed = np.asarray(start_q, dtype=float).copy()
        dq = np.asarray(start_dq, dtype=float).copy()
        if seed.shape != (14,) or dq.shape != (14,) or not np.all(np.isfinite((seed, dq))):
            raise ValueError("measured arm state is invalid")
        snapshot = self._snapshot()
        waypoints: list[ArmWaypoint] = []
        maximum_jump = maximum_position = maximum_orientation = 0.0
        try:
            for index in range(1, count + FILTER_SETTLE_WAYPOINTS + 1):
                alpha = min(index / count, 1.0)
                left = start_left.copy()
                right = start_right.copy()
                left[:3, 3] = (1 - alpha) * start_left[:3, 3] + alpha * end_left[:3, 3]
                right[:3, 3] = (1 - alpha) * start_right[:3, 3] + alpha * end_right[:3, 3]
                left[:3, :3] = slerp(start_left[:3, :3], end_left[:3, :3], alpha)
                right[:3, :3] = slerp(start_right[:3, :3], end_right[:3, :3], alpha)
                posture_seed = self._posture_seed(seed, moving_side, elbow)
                q, tau, raw, jump = self._checked_ik(left, right, posture_seed, seed, dq, index)
                solved_left, solved_right = self.wrist_poses(raw)
                position = max(float(np.linalg.norm(solved_left[:3, 3] - left[:3, 3])),
                               float(np.linalg.norm(solved_right[:3, 3] - right[:3, 3])))
                orientation = max(rotation_error_rad(solved_left[:3, :3], left[:3, :3]),
                                  rotation_error_rad(solved_right[:3, :3], right[:3, :3]))
                if position > MAX_POSITION_ERROR_M or orientation > MAX_ORIENTATION_ERROR_RAD:
                    raise RuntimeError(f"IK residual at waypoint {index}: {position:.4f} m, {np.rad2deg(orientation):.2f} deg")
                waypoints.append(ArmWaypoint(q, tau))
                maximum_jump = max(maximum_jump, jump)
                maximum_position = max(maximum_position, position)
                maximum_orientation = max(maximum_orientation, orientation)
                seed, dq = q, np.zeros(14)
        except Exception:
            self._restore(snapshot)
            raise
        return PlannedTrajectory(tuple(waypoints), duration_s, end_left.copy(), end_right.copy(),
                                 maximum_jump, maximum_position, maximum_orientation)
