"""Small, hardware-independent value types shared by the control stack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ArmSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class ElbowBias(str, Enum):
    AUTO = "auto"
    DOWN = "down"
    OUT = "out"
    IN = "in"
    UP = "up"


@dataclass(frozen=True)
class PoseTarget:
    side: ArmSide
    xyz: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    duration_s: float
    elbow: ElbowBias = ElbowBias.AUTO

    def validate(self) -> None:
        values = (*self.xyz, *self.rpy_deg, self.duration_s)
        if not np.all(np.isfinite(values)):
            raise ValueError("pose target values must be finite")
        if not 0.05 <= self.duration_s <= 60.0:
            raise ValueError("duration_s must be in [0.05, 60]")


@dataclass(frozen=True)
class MarkerOffset:
    """Wrist pose relative to an observed marker frame."""

    xyz_m: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]

    def validate(self) -> None:
        if not np.all(np.isfinite((*self.xyz_m, *self.rpy_deg))):
            raise ValueError("ArUco offset values must be finite")
        if np.linalg.norm(self.xyz_m) > 0.5:
            raise ValueError("ArUco offset must be within 0.5 m of the marker")


@dataclass(frozen=True)
class RobotVisualizationState:
    """Read-only state copied for visualization consumers."""

    body_q: np.ndarray
    hands: dict[str, dict[str, float] | None]
    planned_arm_q: np.ndarray | None
    aruco_target: dict[str, object] | None
