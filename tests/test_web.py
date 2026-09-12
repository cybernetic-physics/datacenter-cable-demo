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
        self.aruco_commands = []

    def start(self): pass
    def close(self): pass
    def attach(self, owner):
        if self.owner is not None: return False
        self.owner = owner; return True
    def detach(self, owner): self.detached = True; self.owner = None
    def stop_motion(self, owner): pass
    def release_control(self, owner): pass
    def command_aruco(self, *args): self.aruco_commands.append(args)
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


class StubSimulation:
    def __init__(self): self.closed = False; self.retries = 0; self.view_updates = []
    def status(self): return {"available": True, "state": "live", "error": None}
    def stream(self): return iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\nSIM\r\n"])
    def retry(self): self.retries += 1; return self.status()
    def update_view(self, *args): self.view_updates.append(args); return {"azimuth_deg": 140}
    def close(self): self.closed = True


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
        self.simulation = StubSimulation()
        self.client_context = TestClient(
            create_app(self.control, self.camera, self.aruco, self.simulation),
            base_url="http://127.0.0.1",
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

    def test_simulation_status_stream_and_retry(self):
        self.assertTrue(self.client.get("/api/simulation/status").json()["available"])
        response = self.client.get("/api/simulation.mjpg")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SIM", response.content)
        self.assertEqual(self.client.post("/api/simulation/retry").status_code, 200)
        self.assertEqual(self.simulation.retries, 1)
        view = self.client.post(
            "/api/simulation/view",
            json={"azimuth_delta_deg": 5, "elevation_delta_deg": -2, "zoom_factor": 1.1},
        )
        self.assertEqual(view.json()["azimuth_deg"], 140)
        self.assertEqual(self.simulation.view_updates[-1], (5.0, -2.0, 1.1, False))

    def test_aruco_config_detections_and_stream(self):
        config = self.client.get("/api/aruco/config").json()
        self.assertEqual(config["dictionary"], "DICT_4X4_50")
        self.assertEqual(config["targeting"]["default_offset"]["xyz_m"], [0.0, 0.0, 0.08])
        self.assertEqual(config["targeting"]["default_offset"]["rpy_deg"], [0.0, 90.0, 90.0])
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

    def test_websocket_accepts_aruco_pose(self):
        with self.client.websocket_connect(
            "/api/control", headers={"origin": "http://127.0.0.1", "host": "127.0.0.1"}
        ) as socket:
            socket.receive_json()
            socket.send_json({
                "type": "aruco_pose",
                "side": "right",
                "marker_id": 3,
                "duration_s": 3.0,
            })
            deadline = time.monotonic() + 1.0
            while not self.control.aruco_commands and time.monotonic() < deadline:
                socket.receive_json()
        self.assertEqual(self.control.aruco_commands[0][1:4], ("right", 3, None))


if __name__ == "__main__":
    unittest.main()
