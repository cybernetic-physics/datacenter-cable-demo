import tempfile
import unittest
from pathlib import Path

import yaml

from classic_control.grasps import (
    Grasp,
    GraspStore,
    hardware_to_logical,
    logical_to_hardware,
)

OPEN = {
    "thumb_rotate": 0.0, "thumb_1": 0.0, "thumb_2": 0.0,
    "index_0": 0.0, "index_1": 0.0, "middle_0": 0.0, "middle_1": 0.0,
}


class GraspTest(unittest.TestCase):
    def test_left_wire_order_round_trip(self):
        logical = {**OPEN, "index_0": -0.2, "middle_0": -0.8}
        wire = logical_to_hardware("left", logical)
        self.assertEqual(wire[3], -0.8)
        self.assertEqual(wire[5], -0.2)
        self.assertEqual(hardware_to_logical("left", wire), logical)

    def test_store_validates_and_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grasps.yaml"
            path.write_text(yaml.safe_dump({"version": 1, "grasps": {
                "open": {"description": "", "left": OPEN, "right": OPEN},
            }}))
            store = GraspStore(path)
            store.save(Grasp("test", "test grasp", {"right": OPEN}))
            self.assertIn("test", store.load())
            store.delete("test")
            self.assertNotIn("test", store.load())
            with self.assertRaises(ValueError):
                store.delete("open")

    def test_out_of_range_is_rejected(self):
        invalid = {**OPEN, "thumb_2": 0.1}
        with self.assertRaises(ValueError):
            logical_to_hardware("right", invalid)


if __name__ == "__main__":
    unittest.main()
