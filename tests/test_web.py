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


class StubCamera:
    def __init__(self): self.closed = False
    def close(self): self.closed = True
    def status(self):
        return {"default": "internal", "sources": {"internal": {
            "name": "Internal camera", "available": True, "streaming": False,
            "location": "test:55555", "resolution": [640, 480], "error": None,
        }}}
    def stream(self, source_id):
        if source_id != "internal": raise KeyError(source_id)
        return iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\nJPEG\r\n"])


class StubAruco:
    def __init__(self):
        self.dictionary = "DICT_4X4_50"
        self.marker_length_mm = None

    def configuration(self):
        return {
            "dictionaries": ["DICT_4X4_50", "DICT_5X5_100"],
            "dictionary": self.dictionary,
            "marker_length_mm": self.marker_length_mm,
            "active": False,
            "calibration": {"valid": True, "expected": {}, "actual": {}, "error": None},
        }

    def configure(self, dictionary, marker_length_mm):
        if dictionary not in self.configuration()["dictionaries"]:
            raise ValueError("unsupported ArUco dictionary")
        self.dictionary = dictionary
        self.marker_length_mm = marker_length_mm
        return self.configuration()

    def latest(self):
        return {"observed_at": None, "markers": []}

    def annotated_stream(self):
        return iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\nARUCO\r\n"])


class WebTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "grasps.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "grasps": {
            "open": {"description": "", "left": OPEN, "right": OPEN},
        }}))
        self.control = StubControl(GraspStore(path))
        self.camera = StubCamera()
        self.aruco = StubAruco()
        self.client_context = TestClient(
            create_app(self.control, self.camera, self.aruco), base_url="http://127.0.0.1"
        )
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

    def test_camera_status_and_stream(self):
        status = self.client.get("/api/cameras").json()
        self.assertTrue(status["sources"]["internal"]["available"])
        response = self.client.get("/api/cameras/internal.mjpg")
        self.assertEqual(response.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", response.headers["content-type"])
        self.assertIn(b"JPEG", response.content)
        self.assertEqual(self.client.get("/api/cameras/missing.mjpg").status_code, 404)

    def test_aruco_config_detections_and_stream(self):
        config = self.client.get("/api/aruco/config").json()
        self.assertEqual(config["dictionary"], "DICT_4X4_50")
        response = self.client.put(
            "/api/aruco/config",
            json={"dictionary": "DICT_5X5_100", "marker_length_mm": 42.0},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["marker_length_mm"], 42.0)
        self.assertEqual(self.client.get("/api/aruco/detections").json()["markers"], [])
        stream = self.client.get("/api/cameras/internal/aruco.mjpg")
        self.assertEqual(stream.status_code, 200)
        self.assertIn(b"ARUCO", stream.content)
        invalid = self.client.put(
            "/api/aruco/config",
            json={"dictionary": "missing", "marker_length_mm": None},
        )
        self.assertEqual(invalid.status_code, 422)

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
