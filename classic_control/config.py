"""Runtime configuration sourced from environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def xr_teleoperate_root() -> Path:
    value = os.environ.get("XR_TELEOPERATE_ROOT")
    if not value:
        raise RuntimeError("XR_TELEOPERATE_ROOT must point to the xr_teleoperate checkout")
    root = Path(value).expanduser().resolve()
    required = (
        root / "teleop/robot_control/robot_arm_ik.py",
        root / "teleop/robot_control/robot_hand_unitree.py",
        root / "assets/g1/g1_body29_hand14.urdf",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"XR_TELEOPERATE_ROOT is not a compatible checkout: {root}")
    return root


def grasp_file() -> Path:
    configured = os.environ.get("CLASSIC_CONTROL_GRASPS")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "config" / "grasps.yaml"


def configure_dds_interface() -> None:
    interface = os.environ.get("ROBOT_NETWORK_INTERFACE")
    if interface and "CYCLONEDDS_URI" not in os.environ:
        os.environ["CYCLONEDDS_URI"] = (
            "<CycloneDDS><Domain><General><Interfaces>"
            f'<NetworkInterface name="{interface}"/>'
            "</Interfaces></General></Domain></CycloneDDS>"
        )
