"""Read-only MuJoCo rendering for robot and ArUco debugging."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .aruco_targeting import (
    ArucoTargeting,
    VisualizedGate,
    VisualizedMarker,
    gate_local_corners,
    pose_transform,
)
from .camera import multipart_jpeg
from .models import RobotVisualizationState

BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HAND_JOINT_NAMES = {
    side: {
        "thumb_rotate": f"{side}_hand_thumb_0_joint",
        "thumb_1": f"{side}_hand_thumb_1_joint",
        "thumb_2": f"{side}_hand_thumb_2_joint",
        "index_0": f"{side}_hand_index_0_joint",
        "index_1": f"{side}_hand_index_1_joint",
        "middle_0": f"{side}_hand_middle_0_joint",
        "middle_1": f"{side}_hand_middle_1_joint",
    }
    for side in ("left", "right")
}

GHOST_BODY_PARTS = ("shoulder", "elbow", "wrist", "hand")
DEFAULT_CAMERA_AZIMUTH = 135.0
DEFAULT_CAMERA_ELEVATION = -20.0
DEFAULT_CAMERA_DISTANCE = 1.8


class VisualizationStateProvider(Protocol):
    def visualization_snapshot(self) -> RobotVisualizationState: ...


def joint_value_map(state: RobotVisualizationState) -> dict[str, float]:
    body_q = np.asarray(state.body_q, dtype=float)
    if body_q.ndim != 1 or len(body_q) < len(BODY_JOINT_NAMES):
        raise ValueError("visualization body state must contain at least 29 joints")
    if not np.all(np.isfinite(body_q[: len(BODY_JOINT_NAMES)])):
        raise ValueError("visualization body state is not finite")
    values = dict(zip(BODY_JOINT_NAMES, body_q[: len(BODY_JOINT_NAMES)]))
    for side, hand in state.hands.items():
        if hand is None or side not in HAND_JOINT_NAMES:
            continue
        for logical_name, joint_name in HAND_JOINT_NAMES[side].items():
            value = float(hand[logical_name])
            if not np.isfinite(value):
                raise ValueError(f"visualization {side} hand state is not finite")
            values[joint_name] = value
    return values


def world_from_base(world_base: np.ndarray, base_object: np.ndarray) -> np.ndarray:
    first = np.asarray(world_base, dtype=float)
    second = np.asarray(base_object, dtype=float)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("visualization poses must be 4x4 transforms")
    if not np.all(np.isfinite((first, second))):
        raise ValueError("visualization poses must be finite")
    return first @ second


class MuJoCoDebugView:
    """Render one shared MJPEG stream without stepping physics or commanding hardware."""

    def __init__(
        self,
        state_provider: VisualizationStateProvider,
        targeting: ArucoTargeting,
        model_path: Path | None,
        width: int = 800,
        height: int = 600,
        fps: float = 12.0,
    ):
        self.state_provider = state_provider
        self.targeting = targeting
        self.model_path = model_path
        self.width = width
        self.height = height
        self.fps = fps
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._error: str | None = None
        self._latest_jpeg: bytes | None = None
        self._sequence = 0
        self._subscribers = 0
        self._camera_azimuth = DEFAULT_CAMERA_AZIMUTH
        self._camera_elevation = DEFAULT_CAMERA_ELEVATION
        self._camera_distance = DEFAULT_CAMERA_DISTANCE

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "available": self._state == "live",
                "state": self._state,
                "resolution": [self.width, self.height],
                "fps": self.fps,
                "error": self._error,
                "subscribers": self._subscribers,
                "view": self._camera_view(),
            }

    def update_view(
        self,
        azimuth_delta_deg: float = 0.0,
        elevation_delta_deg: float = 0.0,
        zoom_factor: float = 1.0,
        reset: bool = False,
    ) -> dict[str, float]:
        """Adjust the shared debug camera without changing simulated or robot state."""
        values = np.asarray(
            (azimuth_delta_deg, elevation_delta_deg, zoom_factor), dtype=float
        )
        if not np.all(np.isfinite(values)) or zoom_factor <= 0:
            raise ValueError("simulation view adjustment is invalid")
        with self._condition:
            if reset:
                self._camera_azimuth = DEFAULT_CAMERA_AZIMUTH
                self._camera_elevation = DEFAULT_CAMERA_ELEVATION
                self._camera_distance = DEFAULT_CAMERA_DISTANCE
            else:
                self._camera_azimuth = (
                    self._camera_azimuth + azimuth_delta_deg + 180.0
                ) % 360.0 - 180.0
                self._camera_elevation = float(
                    np.clip(self._camera_elevation + elevation_delta_deg, -89.0, 89.0)
                )
                self._camera_distance = float(
                    np.clip(self._camera_distance * zoom_factor, 0.35, 4.0)
                )
            return self._camera_view()

    def _camera_view(self) -> dict[str, float]:
        return {
            "azimuth_deg": self._camera_azimuth,
            "elevation_deg": self._camera_elevation,
            "distance_m": self._camera_distance,
        }

    def stream(self) -> Iterator[bytes]:
        self._ensure_started()
        return self._stream_frames()

    def retry(self) -> dict[str, object]:
        self._ensure_started(retry=True)
        return self.status()

    def _ensure_started(self, retry: bool = False) -> None:
        with self._condition:
            if self._stop.is_set():
                raise RuntimeError("MuJoCo debug view is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            if self._state == "error" and not retry:
                return
            self._state = "starting"
            self._error = None
            self._latest_jpeg = None
            self._thread = threading.Thread(
                target=self._render_loop, name="mujoco-debug-view", daemon=True
            )
            self._thread.start()

    def _stream_frames(self) -> Iterator[bytes]:
        sequence = -1
        with self._condition:
            self._subscribers += 1
            self._condition.notify_all()
        try:
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda previous=sequence: self._sequence != previous
                        or self._state in {"error", "closed"},
                        timeout=1.0,
                    )
                    if self._state in {"error", "closed"} and self._latest_jpeg is None:
                        return
                    jpeg = self._latest_jpeg
                    sequence = self._sequence
                    terminal = self._state in {"error", "closed"}
                if jpeg is not None:
                    yield multipart_jpeg(jpeg)
                if terminal:
                    return
        finally:
            with self._condition:
                self._subscribers -= 1
                self._condition.notify_all()

    def _render_loop(self) -> None:
        renderer = None
        try:
            if self.model_path is None:
                raise RuntimeError("XR_TELEOPERATE_ROOT is unavailable for the MuJoCo model")
            if not self.model_path.is_file():
                raise RuntimeError(f"MuJoCo model not found: {self.model_path}")
            os.environ.setdefault("MUJOCO_GL", "egl")
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self._show_floor_with_visual_geoms(mujoco, model)
            model.light_castshadow[:] = 0
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
            measured_data = mujoco.MjData(model)
            target_data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, self.height, self.width)
            camera = self._camera(mujoco)
            option = mujoco.MjvOption()
            option.geomgroup[:] = 0
            option.geomgroup[1] = 1
            qpos_addresses = self._qpos_addresses(mujoco, model)
            ghost_geoms = self._ghost_geoms(mujoco, model)
            period = 1.0 / self.fps
            deadline = time.monotonic()
            with self._condition:
                self._state = "live"
                self._condition.notify_all()
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._subscribers > 0 or self._stop.is_set(),
                        timeout=1.0,
                    )
                    if self._stop.is_set():
                        break
                    if self._subscribers == 0:
                        continue
                deadline = max(deadline, time.monotonic())
                state_error = None
                try:
                    snapshot = self.state_provider.visualization_snapshot()
                except Exception as error:
                    snapshot = None
                    state_error = str(error)
                measured_data.qpos[:] = model.qpos0
                if snapshot is not None:
                    self._set_joint_values(measured_data, qpos_addresses, joint_value_map(snapshot))
                mujoco.mj_forward(model, measured_data)
                target_ready = bool(snapshot is not None and snapshot.planned_arm_q is not None)
                if target_ready:
                    target_data.qpos[:] = measured_data.qpos
                    planned = np.asarray(snapshot.planned_arm_q, dtype=float)
                    if planned.shape != (14,) or not np.all(np.isfinite(planned)):
                        target_ready = False
                    else:
                        target_values = dict(zip(BODY_JOINT_NAMES[15:29], planned))
                        self._set_joint_values(target_data, qpos_addresses, target_values)
                        mujoco.mj_forward(model, target_data)
                markers = self.targeting.visualized_markers()
                gates = self.targeting.visualized_gates()
                world_base = self._world_base(mujoco, model, measured_data)
                aruco_target = None if snapshot is None else snapshot.aruco_target
                selected_gate = (
                    aruco_target.get("gate_index")
                    if aruco_target is not None
                    and aruco_target.get("target_kind") == "gate"
                    else None
                )
                self._apply_camera_view(camera)
                renderer.update_scene(measured_data, camera=camera, scene_option=option)
                if target_ready:
                    self._add_ghost_geoms(mujoco, renderer.scene, model, target_data, ghost_geoms)
                for marker in markers:
                    self._add_marker(mujoco, renderer.scene, world_base, marker)
                for gate in gates:
                    self._add_gate(
                        mujoco,
                        renderer.scene,
                        world_base,
                        gate,
                        gate.gate_index == selected_gate,
                    )
                if aruco_target is not None:
                    self._add_target(mujoco, renderer.scene, world_base, aruco_target)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                rgb = renderer.render()
                jpeg = self._encode_frame(rgb, markers, aruco_target, state_error, target_ready)
                with self._condition:
                    self._latest_jpeg = jpeg
                    self._sequence += 1
                    self._condition.notify_all()
                deadline += period
                self._stop.wait(max(0.0, deadline - time.monotonic()))
        except Exception as error:
            with self._condition:
                self._state = "error"
                self._error = str(error)
                self._condition.notify_all()
        finally:
            if renderer is not None:
                renderer.close()

    def _camera(self, mujoco):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.15, 0.0, 0.8)
        self._apply_camera_view(camera)
        return camera

    def _apply_camera_view(self, camera) -> None:
        with self._condition:
            camera.azimuth = self._camera_azimuth
            camera.elevation = self._camera_elevation
            camera.distance = self._camera_distance

    @staticmethod
    def _qpos_addresses(mujoco, model) -> dict[str, int]:
        addresses: dict[str, int] = {}
        for name in (*BODY_JOINT_NAMES, *HAND_JOINT_NAMES["left"].values(), *HAND_JOINT_NAMES["right"].values()):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"MuJoCo model is missing joint {name}")
            addresses[name] = int(model.jnt_qposadr[joint_id])
        return addresses

    @staticmethod
    def _show_floor_with_visual_geoms(mujoco, model) -> None:
        """Expose only the floor from collision group 0 in the rendered group 1."""
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0:
            raise RuntimeError("MuJoCo model is missing the floor geom")
        model.geom_group[floor_id] = 1

    @staticmethod
    def _set_joint_values(data, addresses: dict[str, int], values: dict[str, float]) -> None:
        for name, value in values.items():
            if name in addresses:
                data.qpos[addresses[name]] = value

    @staticmethod
    def _ghost_geoms(mujoco, model) -> tuple[int, ...]:
        selected: list[int] = []
        for geom_id in range(model.ngeom):
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
            ) or ""
            if int(model.geom_group[geom_id]) == 1 and any(
                part in body_name for part in GHOST_BODY_PARTS
            ):
                selected.append(geom_id)
        return tuple(selected)

    @staticmethod
    def _append_geom(mujoco, scene, geom_type, size, pos, mat, rgba):
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            geom,
            int(geom_type),
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.asarray(mat, dtype=np.float64).reshape(9),
            np.asarray(rgba, dtype=np.float32),
        )
        return geom

    def _add_ghost_geoms(self, mujoco, scene, model, data, geom_ids: tuple[int, ...]) -> None:
        for geom_id in geom_ids:
            geom = self._append_geom(
                mujoco,
                scene,
                model.geom_type[geom_id],
                model.geom_size[geom_id],
                data.geom_xpos[geom_id],
                data.geom_xmat[geom_id],
                (0.1, 0.85, 1.0, 0.26),
            )
            if geom is not None:
                geom.dataid = int(model.geom_dataid[geom_id])
                geom.emission = 0.25
                geom.transparent = 1

    @staticmethod
    def _world_base(mujoco, model, data) -> np.ndarray:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        transform = np.eye(4)
        transform[:3, :3] = data.xmat[pelvis_id].reshape(3, 3)
        transform[:3, 3] = data.xpos[pelvis_id]
        return transform

    def _add_marker(self, mujoco, scene, world_base_pose: np.ndarray, marker: VisualizedMarker) -> None:
        pose = world_from_base(world_base_pose, marker.base_from_marker)
        geom = self._append_geom(
            mujoco,
            scene,
            mujoco.mjtGeom.mjGEOM_BOX,
            (marker.size_m / 2, marker.size_m / 2, 0.002),
            pose[:3, 3],
            pose[:3, :3],
            (1.0, 0.65, 0.08, 0.82),
        )
        if geom is not None:
            geom.emission = 0.25
        self._add_axes(mujoco, scene, pose, max(0.05, marker.size_m))

    def _add_gate(
        self,
        mujoco,
        scene,
        world_base_pose: np.ndarray,
        gate: VisualizedGate,
        selected: bool,
    ) -> None:
        pose = world_from_base(world_base_pose, gate.base_from_gate)
        color = (1.0, 0.15, 0.65, 1.0) if selected else (0.2, 1.0, 0.4, 0.9)
        local_corners = gate_local_corners()
        corners = (pose[:3, :3] @ local_corners.T).T + pose[:3, 3]
        for index in range(4):
            self._add_connector(
                mujoco,
                scene,
                corners[index],
                corners[(index + 1) % 4],
                color,
                4.0 if selected else 2.0,
            )
        geom = self._append_geom(
            mujoco,
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (0.007 if selected else 0.003,) * 3,
            pose[:3, 3],
            pose[:3, :3],
            color,
        )
        if geom is not None:
            geom.emission = 0.35

    def _add_target(self, mujoco, scene, world_base_pose: np.ndarray, target: dict[str, object]) -> None:
        marker = target.get("base_marker")
        anchor = target.get("base_gate", marker)
        wrist = target.get("base_wrist_target")
        if not isinstance(anchor, dict) or not isinstance(wrist, dict):
            return
        try:
            anchor_pose = pose_transform(tuple(anchor["xyz"]), tuple(anchor["rpy_deg"]))
            wrist_pose = pose_transform(tuple(wrist["xyz"]), tuple(wrist["rpy_deg"]))
        except (KeyError, TypeError, ValueError):
            return
        world_anchor = world_from_base(world_base_pose, anchor_pose)
        world_wrist = world_from_base(world_base_pose, wrist_pose)
        self._add_axes(mujoco, scene, world_wrist, 0.10)
        self._add_connector(
            mujoco,
            scene,
            world_anchor[:3, 3],
            world_wrist[:3, 3],
            (0.2, 0.9, 1.0, 0.9),
            3.0,
        )

    def _add_axes(self, mujoco, scene, pose: np.ndarray, length: float) -> None:
        colors = ((1.0, 0.15, 0.1, 1.0), (0.15, 1.0, 0.2, 1.0), (0.2, 0.45, 1.0, 1.0))
        for axis, color in enumerate(colors):
            self._add_connector(
                mujoco,
                scene,
                pose[:3, 3],
                pose[:3, 3] + pose[:3, axis] * length,
                color,
                4.0,
            )

    def _add_connector(self, mujoco, scene, start, end, rgba, width: float) -> None:
        geom = self._append_geom(
            mujoco,
            scene,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.ones(3),
            np.zeros(3),
            np.eye(3),
            rgba,
        )
        if geom is not None:
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                width,
                np.asarray(start, dtype=np.float64),
                np.asarray(end, dtype=np.float64),
            )

    @staticmethod
    def _pose_text(name: str, pose: object) -> str | None:
        if not isinstance(pose, dict):
            return None
        try:
            xyz = " ".join(f"{float(value):+.3f}" for value in pose["xyz"])
            rpy = " ".join(f"{float(value):+.1f}" for value in pose["rpy_deg"])
        except (KeyError, TypeError, ValueError):
            return None
        return f"{name} XYZ {xyz} m   RPY {rpy} deg"

    def _encode_frame(
        self,
        rgb: np.ndarray,
        markers: tuple[VisualizedMarker, ...],
        target: dict[str, object] | None,
        state_error: str | None,
        target_ready: bool,
    ) -> bytes:
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lines = ["MEASURED ROBOT  |  CYAN = PLANNED ARM"]
        if state_error:
            lines.append(f"Robot state unavailable: {state_error}")
        if markers:
            lines.extend(
                f"ArUco {marker.marker_id}: SAVED {marker.saved_for_s:.1f}s, "
                f"capture error {marker.reprojection_error_px:.2f}px"
                for marker in markers
            )
        else:
            lines.append("No session-saved metric ArUco markers")
        if target is not None:
            if target.get("target_kind") == "gate":
                lines.append(f"Selected Ethernet gate {target.get('gate_index')} (magenta)")
            marker_text = self._pose_text("Marker", target.get("base_marker"))
            gate_text = self._pose_text("Gate center", target.get("base_gate"))
            wrist_text = self._pose_text("Wrist target", target.get("base_wrist_target"))
            lines.extend(text for text in (marker_text, gate_text, wrist_text) if text)
            if not target_ready:
                lines.append("No valid IK ghost for this target")
        for index, line in enumerate(lines):
            color = (90, 220, 255) if index else (255, 255, 255)
            cv2.putText(
                image,
                line,
                (16, 28 + 24 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                line,
                (16, 28 + 24 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                1,
                cv2.LINE_AA,
            )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("could not encode MuJoCo debug frame")
        return encoded.tobytes()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._condition:
            self._state = "closed"
            self._condition.notify_all()
