"""The sole DDS owner for G1 body and dual Dex3 commands."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import configure_dds_interface, unitree_sdk2py_root
from .grasps import hardware_to_logical, logical_to_hardware

BODY_MOTOR_COUNT = 35
ARM_INDICES = tuple(range(15, 29))
WAIST_INDICES = (12, 13, 14)
STATE_STALE_S = 0.5


@dataclass(frozen=True)
class RobotState:
    body_q: np.ndarray
    body_dq: np.ndarray
    hands: dict[str, dict[str, float] | None]
    acquired: bool
    fault: str | None = None


class RobotBackend(Protocol):
    def connect(self) -> None: ...
    def state(self) -> RobotState: ...
    def acquire(self) -> None: ...
    def set_body_target(self, arm_q: np.ndarray, arm_tau: np.ndarray, waist_q: np.ndarray | None = None) -> None: ...
    def set_hand_target(self, side: str, joints: dict[str, float]) -> None: ...
    def release(self) -> None: ...
    def close(self) -> None: ...


class UnitreeRobotBackend:
    """Own exactly one body command stream and both Dex3 command streams."""

    def __init__(self, xr_root: Path, body_hz: float = 250.0, hand_hz: float = 40.0,
                 state_timeout_s: float = 5.0):
        self.xr_root = xr_root
        self.body_period = 1.0 / body_hz
        self.hand_period = 1.0 / hand_hz
        self.state_timeout_s = state_timeout_s
        self._lock = threading.RLock()
        self._body_q: np.ndarray | None = None
        self._body_dq: np.ndarray | None = None
        self._body_updated = 0.0
        self._mode_machine: int | None = None
        self._hands_raw: dict[str, list[float] | None] = {"left": None, "right": None}
        self._hands_updated = {"left": 0.0, "right": 0.0}
        self._hand_targets: dict[str, dict[str, float] | None] = {"left": None, "right": None}
        self._arm_q: np.ndarray | None = None
        self._arm_tau = np.zeros(14)
        self._waist_q: np.ndarray | None = None
        self._acquired = False
        self._closed = False
        self._fault: str | None = None
        self._publish_stop: threading.Event | None = None
        self._publish_threads: list[threading.Thread] = []

    def connect(self) -> None:
        configure_dds_interface()
        sys.path.insert(0, str(self.xr_root))
        sdk_root = unitree_sdk2py_root()
        if sdk_root is not None:
            sys.path.insert(0, str(sdk_root))
        from teleop.utils.motion_switcher import MotionSwitcher
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            HandCmd_,
            HandState_,
            LowCmd_,
            LowState_,
        )
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0)
        self._ChannelPublisher = ChannelPublisher
        self._lowcmd_type = LowCmd_
        self._lowcmd_factory = unitree_hg_msg_dds__LowCmd_
        self._handcmd_type = HandCmd_
        self._handcmd_factory = unitree_hg_msg_dds__HandCmd_
        self._crc = CRC()
        self._motion_switcher = MotionSwitcher()

        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init()
        self._hand_subs = {}
        for side in ("left", "right"):
            subscriber = ChannelSubscriber(f"rt/dex3/{side}/state", HandState_)
            subscriber.Init()
            self._hand_subs[side] = subscriber
        threading.Thread(target=self._state_loop, name="robot-state", daemon=True).start()

        deadline = time.monotonic() + self.state_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._body_q is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError("no finite rt/lowstate received")

    def _state_loop(self) -> None:
        while not self._closed:
            message = self._lowstate_sub.Read()
            if message is not None:
                q = np.array([message.motor_state[index].q for index in range(BODY_MOTOR_COUNT)])
                dq = np.array([message.motor_state[index].dq for index in range(BODY_MOTOR_COUNT)])
                if np.all(np.isfinite(q)) and np.all(np.isfinite(dq)):
                    with self._lock:
                        self._body_q, self._body_dq = q, dq
                        self._mode_machine = message.mode_machine
                        self._body_updated = time.monotonic()
            for side, subscriber in self._hand_subs.items():
                hand = subscriber.Read()
                if hand is not None:
                    values = [float(hand.motor_state[index].q) for index in range(7)]
                    if np.all(np.isfinite(values)):
                        with self._lock:
                            self._hands_raw[side] = values
                            self._hands_updated[side] = time.monotonic()
            time.sleep(0.002)

    def state(self) -> RobotState:
        with self._lock:
            if self._body_q is None or self._body_dq is None:
                raise RuntimeError("body state is not ready")
            fault = self._fault
            if time.monotonic() - self._body_updated > STATE_STALE_S:
                fault = "rt/lowstate is stale"
            hands = {
                side: None if values is None else hardware_to_logical(side, values)
                for side, values in self._hands_raw.items()
            }
            return RobotState(self._body_q.copy(), self._body_dq.copy(), hands, self._acquired, fault)

    @staticmethod
    def _gains(index: int) -> tuple[float, float]:
        if index in ARM_INDICES:
            return (40.0, 1.5) if index in (19, 20, 21, 26, 27, 28) else (80.0, 3.0)
        if index in WAIST_INDICES:
            return 100.0, 3.0
        return 300.0, 3.0

    @staticmethod
    def _dex3_mode(index: int) -> int:
        return (index & 0x0F) | (0x01 << 4)

    def acquire(self) -> None:
        with self._lock:
            if self._acquired:
                return
            state = self.state()
        status, _ = self._motion_switcher.Enter_Debug_Mode()
        if status != 0:
            raise RuntimeError(f"could not enter Unitree debug mode (status {status})")
        try:
            body_publisher = self._ChannelPublisher("rt/lowcmd", self._lowcmd_type)
            body_publisher.Init()
            hand_publishers = {}
            for side in ("left", "right"):
                publisher = self._ChannelPublisher(f"rt/dex3/{side}/cmd", self._handcmd_type)
                publisher.Init()
                hand_publishers[side] = publisher
            message = self._lowcmd_factory()
            message.mode_pr = 0
            message.mode_machine = self._mode_machine
            for index in range(BODY_MOTOR_COUNT):
                command = message.motor_cmd[index]
                command.mode = 1
                command.q = float(state.body_q[index])
                command.dq = 0.0
                command.tau = 0.0
                command.kp, command.kd = self._gains(index)
            with self._lock:
                self._body_publisher = body_publisher
                self._hand_publishers = hand_publishers
                self._body_message = message
                self._arm_q = state.body_q[list(ARM_INDICES)].copy()
                self._arm_tau = np.zeros(14)
                self._waist_q = state.body_q[list(WAIST_INDICES)].copy()
                for side in ("left", "right"):
                    if state.hands[side] is not None:
                        self._hand_targets[side] = state.hands[side]
                self._fault = None
                self._publish_stop = threading.Event()
                self._acquired = True
            self._publish_threads = [
                threading.Thread(target=self._body_publish_loop, name="body-command", daemon=True),
                threading.Thread(target=self._hand_publish_loop, name="hand-command", daemon=True),
            ]
            for thread in self._publish_threads:
                thread.start()
        except Exception:
            self._motion_switcher.Exit_Debug_Mode()
            raise

    def _record_publish_fault(self, error: Exception) -> None:
        with self._lock:
            self._fault = f"command publisher failed: {error}"
            if self._publish_stop is not None:
                self._publish_stop.set()

    def _body_publish_loop(self) -> None:
        deadline = time.monotonic()
        try:
            while self._publish_stop is not None and not self._publish_stop.is_set():
                with self._lock:
                    if time.monotonic() - self._body_updated > STATE_STALE_S:
                        raise RuntimeError("rt/lowstate is stale")
                    arm_q = self._arm_q.copy()
                    arm_tau = self._arm_tau.copy()
                    waist_q = self._waist_q.copy()
                    message = self._body_message
                for offset, motor_index in enumerate(WAIST_INDICES):
                    message.motor_cmd[motor_index].q = float(waist_q[offset])
                    message.motor_cmd[motor_index].tau = 0.0
                for offset, motor_index in enumerate(ARM_INDICES):
                    message.motor_cmd[motor_index].q = float(arm_q[offset])
                    message.motor_cmd[motor_index].tau = float(arm_tau[offset])
                message.crc = self._crc.Crc(message)
                self._body_publisher.Write(message)
                deadline += self.body_period
                time.sleep(max(0.0, deadline - time.monotonic()))
        except Exception as error:
            self._record_publish_fault(error)

    def _hand_publish_loop(self) -> None:
        deadline = time.monotonic()
        try:
            while self._publish_stop is not None and not self._publish_stop.is_set():
                with self._lock:
                    targets = {side: None if target is None else target.copy()
                               for side, target in self._hand_targets.items()}
                for side, target in targets.items():
                    if target is None:
                        continue
                    with self._lock:
                        if time.monotonic() - self._hands_updated[side] > STATE_STALE_S:
                            raise RuntimeError(f"rt/dex3/{side}/state is stale")
                    values = logical_to_hardware(side, target)
                    message = self._handcmd_factory()
                    for index, value in enumerate(values):
                        motor = message.motor_cmd[index]
                        motor.mode = self._dex3_mode(index)
                        motor.q = value
                        motor.dq = 0.0
                        motor.tau = 0.0
                        motor.kp = 0.8
                        motor.kd = 0.3
                    self._hand_publishers[side].Write(message)
                deadline += self.hand_period
                time.sleep(max(0.0, deadline - time.monotonic()))
        except Exception as error:
            self._record_publish_fault(error)

    def set_body_target(self, arm_q: np.ndarray, arm_tau: np.ndarray,
                        waist_q: np.ndarray | None = None) -> None:
        q, tau = np.asarray(arm_q, dtype=float), np.asarray(arm_tau, dtype=float)
        if q.shape != (14,) or tau.shape != (14,) or not np.all(np.isfinite((q, tau))):
            raise ValueError("arm target must contain finite 14-element q and tau vectors")
        with self._lock:
            if not self._acquired:
                raise RuntimeError("robot control has not been acquired")
            self._arm_q, self._arm_tau = q.copy(), tau.copy()
            if waist_q is not None:
                waist = np.asarray(waist_q, dtype=float)
                if waist.shape != (3,) or not np.all(np.isfinite(waist)):
                    raise ValueError("waist target must contain three finite values")
                self._waist_q = waist.copy()

    def set_hand_target(self, side: str, joints: dict[str, float]) -> None:
        logical_to_hardware(side, joints)  # validation
        with self._lock:
            if not self._acquired:
                raise RuntimeError("robot control has not been acquired")
            if self._hands_raw[side] is None:
                raise RuntimeError(f"no finite {side} Dex3 state received")
            if time.monotonic() - self._hands_updated[side] > STATE_STALE_S:
                raise RuntimeError(f"rt/dex3/{side}/state is stale")
            self._hand_targets[side] = joints.copy()

    def release(self) -> None:
        with self._lock:
            if not self._acquired:
                return
            stop = self._publish_stop
            self._acquired = False
        if stop is not None:
            stop.set()
        for thread in self._publish_threads:
            thread.join(timeout=1.0)
        self._publish_threads = []
        status, _ = self._motion_switcher.Exit_Debug_Mode()
        if status != 0:
            raise RuntimeError(f"could not return control to Unitree AI mode (status {status})")

    def close(self) -> None:
        try:
            self.release()
        finally:
            self._closed = True
