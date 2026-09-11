"""Named, validated Dex3 grasp profiles with atomic YAML persistence."""

from __future__ import annotations

import math
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

JOINT_NAMES = (
    "thumb_rotate", "thumb_1", "thumb_2",
    "index_0", "index_1", "middle_0", "middle_1",
)

# Logical order above. Hardware order differs for the left hand.
DEX3_LIMITS = {
    "left": {
        "thumb_rotate": (-1.00, 1.00), "thumb_1": (-0.72, 1.04),
        "thumb_2": (0.00, 1.74), "index_0": (-1.57, 0.00),
        "index_1": (-1.74, 0.00), "middle_0": (-1.57, 0.00),
        "middle_1": (-1.74, 0.00),
    },
    "right": {
        "thumb_rotate": (-1.00, 1.00), "thumb_1": (-1.04, 0.72),
        "thumb_2": (-1.74, 0.00), "index_0": (-0.05, 1.57),
        "index_1": (-0.05, 1.74), "middle_0": (-0.05, 1.57),
        "middle_1": (-0.05, 1.74),
    },
}

HARDWARE_ORDER = {
    "left": ("thumb_rotate", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"),
    "right": JOINT_NAMES,
}


def validate_hand(side: str, joints: Mapping[str, float]) -> dict[str, float]:
    if side not in DEX3_LIMITS:
        raise ValueError(f"unknown hand side: {side}")
    if set(joints) != set(JOINT_NAMES):
        missing = sorted(set(JOINT_NAMES) - set(joints))
        extra = sorted(set(joints) - set(JOINT_NAMES))
        raise ValueError(f"invalid {side} joints; missing={missing}, extra={extra}")
    result: dict[str, float] = {}
    for name in JOINT_NAMES:
        value = float(joints[name])
        low, high = DEX3_LIMITS[side][name]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{side}.{name} must be finite and in [{low}, {high}]")
        result[name] = value
    return result


def logical_to_hardware(side: str, joints: Mapping[str, float]) -> list[float]:
    validated = validate_hand(side, joints)
    return [validated[name] for name in HARDWARE_ORDER[side]]


def hardware_to_logical(side: str, values: list[float]) -> dict[str, float]:
    if side not in HARDWARE_ORDER or len(values) != 7:
        raise ValueError("Dex3 state must contain seven values for a known side")
    mapped = dict(zip(HARDWARE_ORDER[side], (float(value) for value in values)))
    return {name: mapped[name] for name in JOINT_NAMES}


@dataclass(frozen=True)
class Grasp:
    name: str
    description: str
    hands: dict[str, dict[str, float]]


class GraspStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _load_unlocked(self) -> dict[str, Grasp]:
        try:
            document = yaml.safe_load(self.path.read_text())
        except FileNotFoundError as error:
            raise RuntimeError(f"grasp file not found: {self.path}") from error
        if not isinstance(document, dict) or document.get("version") != 1:
            raise ValueError("grasp file must have schema version 1")
        raw_grasps = document.get("grasps")
        if not isinstance(raw_grasps, dict):
            raise TypeError("grasps must be a mapping")
        result: dict[str, Grasp] = {}
        for name, raw in raw_grasps.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(raw, dict):
                raise ValueError("each grasp must have a non-empty name and mapping value")
            description = str(raw.get("description", ""))
            hands: dict[str, dict[str, float]] = {}
            for side in ("left", "right"):
                if side in raw:
                    if not isinstance(raw[side], dict):
                        raise ValueError(f"{name}.{side} must be a joint mapping")
                    hands[side] = validate_hand(side, raw[side])
            if not hands:
                raise ValueError(f"grasp {name!r} must define at least one hand")
            result[name] = Grasp(name, description, hands)
        if "open" not in result or set(result["open"].hands) != {"left", "right"}:
            raise ValueError("grasp file must define an open pose for both hands")
        return result

    def load(self) -> dict[str, Grasp]:
        with self._lock:
            return self._load_unlocked()

    def save(self, grasp: Grasp) -> None:
        name = grasp.name.strip()
        if not name:
            raise ValueError("grasp name cannot be empty")
        hands = {side: validate_hand(side, values) for side, values in grasp.hands.items()}
        if not hands:
            raise ValueError("a grasp must define at least one hand")
        with self._lock:
            current = self._load_unlocked()
            current[name] = Grasp(name, grasp.description, hands)
            self._write_unlocked(current)

    def delete(self, name: str) -> None:
        if name == "open":
            raise ValueError("the required open grasp cannot be deleted")
        with self._lock:
            current = self._load_unlocked()
            if name not in current:
                raise KeyError(name)
            del current[name]
            self._write_unlocked(current)

    def _write_unlocked(self, grasps: Mapping[str, Grasp]) -> None:
        document = {"version": 1, "grasps": {}}
        for name, grasp in sorted(grasps.items()):
            item: dict[str, object] = {"description": grasp.description}
            item.update(grasp.hands)
            document["grasps"][name] = item
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                yaml.safe_dump(document, stream, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
