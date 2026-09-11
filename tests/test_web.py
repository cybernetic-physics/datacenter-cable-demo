import tempfile
import time
import unittest
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from classic_control.grasps import GraspStore
from classic_control.web import create_app

OPEN = {
    "thumb_rotate": 0.0, "thumb_1": 0.0, "thumb_2": 0.0,
    "index_0": 0.0, "index_1": 0.0, "middle_0": 0.0, "middle_1": 0.0,
}


class StubControl:
    def __init__(self, store):
        self.grasps = store
        self.owner = None
        self.detached = False

    def start(self): pass
    def close(self): pass
    def attach(self, owner):
        if self.owner is not None: return False
        self.owner = owner; return True
    def detach(self, owner): self.detached = True; self.owner = None
    def stop_motion(self, owner): pass
    def release_control(self, owner): pass
    def telemetry(self):
        return {"connected": True, "owner": self.owner is not None,
                "acquired": False, "active_command": None, "fault": None,
                "arms": None, "hands": None}


class WebTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "grasps.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "grasps": {
            "open": {"description": "", "left": OPEN, "right": OPEN},
        }}))
        self.control = StubControl(GraspStore(path))
        self.client_context = TestClient(create_app(self.control), base_url="http://127.0.0.1")
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_static_page_and_config(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        config = self.client.get("/api/config").json()
        self.assertEqual(len(config["joint_names"]), 7)
        self.assertIn("left", config["dex3_limits"])

    def test_grasp_crud(self):
        response = self.client.put("/api/grasps/test", json={"description": "test", "hands": {"right": OPEN}})
        self.assertEqual(response.status_code, 200)
        self.assertIn("test", self.client.get("/api/grasps").json())
        self.assertEqual(self.client.delete("/api/grasps/test").status_code, 200)

    def test_websocket_streams_state_and_detaches(self):
        with self.client.websocket_connect(
            "/api/control", headers={"origin": "http://127.0.0.1", "host": "127.0.0.1"}
        ) as socket:
            message = socket.receive_json()
            self.assertEqual(message["type"], "telemetry")
            self.assertTrue(message["state"]["connected"])
            socket.close()
        deadline = time.monotonic() + 2.0
        while not self.control.detached and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.control.detached)


if __name__ == "__main__":
    unittest.main()
