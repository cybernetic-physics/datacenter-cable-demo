"""FastAPI application and browser-control protocol."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .arm import ArmPlanner
from .aruco import ArucoVision, draw_projected_gates
from .aruco_targeting import ArucoTargeting
from .camera import CameraBusy, CameraHub, CameraUnavailable
from .config import grasp_file, xr_teleoperate_root
from .grasps import DEX3_LIMITS, JOINT_NAMES, Grasp, GraspStore
from .hardware import UnitreeRobotBackend
from .models import ArmSide, ElbowBias, MarkerOffset, PoseTarget
from .service import ControlService
from .simulation import MuJoCoDebugView

STATIC = Path(__file__).resolve().parent / "static"


class GraspBody(BaseModel):
    description: str = ""
    hands: dict[str, dict[str, float]]


class JogMessage(BaseModel):
    side: ArmSide
    mode: str
    axis: int = Field(ge=0, le=2)
    delta: float
    duration_s: float = Field(default=0.2, ge=0.05, le=2.0)
    elbow: ElbowBias = ElbowBias.AUTO


class PoseMessage(BaseModel):
    side: ArmSide
    xyz: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    duration_s: float = Field(default=3.0, ge=0.05, le=60.0)
    elbow: ElbowBias = ElbowBias.AUTO


class HandMessage(BaseModel):
    targets: dict[str, dict[str, float]]
    duration_s: float = Field(default=0.5, ge=0.0, le=10.0)


class GraspMessage(BaseModel):
    name: str
    sides: list[str] = Field(default_factory=list)
    duration_s: float = Field(default=0.5, ge=0.0, le=10.0)


class ArucoConfigBody(BaseModel):
    dictionary: str
    marker_length_mm: float | None = Field(default=None, gt=0, le=1000)


class ArucoOffsetBody(BaseModel):
    xyz_m: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


class ArucoPoseMessage(BaseModel):
    side: ArmSide
    marker_id: int = Field(ge=0)
    offset: ArucoOffsetBody | None = None
    duration_s: float = Field(default=3.0, ge=0.05, le=60.0)
    elbow: ElbowBias = ElbowBias.AUTO


class GateMessage(BaseModel):
    side: ArmSide
    gate: int = Field(ge=0, le=23)
    offset: ArucoOffsetBody | None = None
    duration_s: float = Field(default=3.0, ge=0.05, le=60.0)
    elbow: ElbowBias = ElbowBias.AUTO


class SimulationViewBody(BaseModel):
    azimuth_delta_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    elevation_delta_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    zoom_factor: float = Field(default=1.0, ge=0.5, le=2.0)
    reset: bool = False


def _real_service(root: Path, targeting: ArucoTargeting) -> ControlService:
    return ControlService(
        UnitreeRobotBackend(root), ArmPlanner(root), GraspStore(grasp_file()), targeting
    )


def create_app(
    control: ControlService | None = None,
    camera: CameraHub | None = None,
    aruco: ArucoVision | None = None,
    simulation: MuJoCoDebugView | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.camera = camera or CameraHub()
        app.state.aruco = aruco or ArucoVision(app.state.camera)
        app.state.aruco_targeting = ArucoTargeting(app.state.aruco)
        if control is None:
            root = xr_teleoperate_root()
            app.state.control = _real_service(root, app.state.aruco_targeting)
        else:
            root = None
            app.state.control = control
        if simulation is None:
            if root is None:
                try:
                    root = xr_teleoperate_root()
                except RuntimeError:
                    pass
            model_path = None if root is None else root / "assets/g1/g1_body29_hand14.xml"
            app.state.simulation = MuJoCoDebugView(
                app.state.control, app.state.aruco_targeting, model_path
            )
        else:
            app.state.simulation = simulation
        await asyncio.to_thread(app.state.control.start)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.simulation.close)
            await asyncio.to_thread(app.state.camera.close)
            await asyncio.to_thread(app.state.control.close)

    app = FastAPI(title="Classic Control", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health():
        return app.state.control.telemetry()

    @app.get("/api/grasps")
    async def list_grasps():
        return {
            name: {"description": grasp.description, "hands": grasp.hands}
            for name, grasp in app.state.control.grasps.load().items()
        }

    @app.get("/api/config")
    async def ui_config():
        return {"joint_names": JOINT_NAMES, "dex3_limits": DEX3_LIMITS}

    @app.get("/api/cameras")
    async def camera_status():
        return await asyncio.to_thread(app.state.camera.status)

    @app.get("/api/simulation/status")
    async def simulation_status():
        return await asyncio.to_thread(app.state.simulation.status)

    @app.post("/api/simulation/retry")
    async def retry_simulation():
        return await asyncio.to_thread(app.state.simulation.retry)

    @app.post("/api/simulation/view")
    async def update_simulation_view(body: SimulationViewBody):
        try:
            return await asyncio.to_thread(
                app.state.simulation.update_view,
                body.azimuth_delta_deg,
                body.elevation_delta_deg,
                body.zoom_factor,
                body.reset,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/simulation.mjpg")
    def simulation_stream():
        try:
            stream = app.state.simulation.stream()
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/cameras/{source_id}.mjpg")
    def camera_stream(source_id: str):
        try:
            stream = app.state.camera.stream(source_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except CameraBusy as error:
            raise HTTPException(409, str(error)) from error
        except CameraUnavailable as error:
            raise HTTPException(503, str(error)) from error
        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/aruco/config")
    async def aruco_config():
        result = await asyncio.to_thread(app.state.aruco.configuration)
        result["targeting"] = app.state.aruco_targeting.configuration()
        return result

    @app.put("/api/aruco/config")
    async def update_aruco_config(body: ArucoConfigBody):
        try:
            result = await asyncio.to_thread(
                app.state.aruco.configure, body.dictionary, body.marker_length_mm
            )
            result["targeting"] = app.state.aruco_targeting.configuration()
            return result
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/aruco/detections")
    async def aruco_detections():
        return await asyncio.to_thread(app.state.aruco.latest)

    @app.get("/api/cameras/internal/aruco.mjpg")
    def aruco_stream():
        def draw_gate_overlay(image, camera_matrix, distortion_coefficients):
            app.state.aruco_targeting.visualized_markers()
            gates = app.state.aruco_targeting.projected_gates(
                camera_matrix, distortion_coefficients
            )
            draw_projected_gates(
                image, gates, app.state.control.selected_gate_index()
            )

        try:
            stream = app.state.aruco.annotated_stream(draw_gate_overlay)
        except CameraBusy as error:
            raise HTTPException(409, str(error)) from error
        except CameraUnavailable as error:
            raise HTTPException(503, str(error)) from error
        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.put("/api/grasps/{name}")
    async def put_grasp(name: str, body: GraspBody):
        try:
            app.state.control.grasps.save(Grasp(name, body.description, body.hands))
            return {"saved": name}
        except (TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.delete("/api/grasps/{name}")
    async def delete_grasp(name: str):
        try:
            app.state.control.grasps.delete(name)
            return {"deleted": name}
        except KeyError as error:
            raise HTTPException(404, f"unknown grasp: {name}") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.websocket("/api/control")
    async def control_socket(socket: WebSocket):
        origin = urlparse(socket.headers.get("origin", ""))
        host = socket.headers.get("host", "")
        if origin.netloc != host or origin.hostname not in {"127.0.0.1", "localhost", "::1"}:
            await socket.close(code=1008, reason="same-origin localhost connection required")
            return
        await socket.accept()
        owner = str(uuid.uuid4())
        service: ControlService = app.state.control
        if not service.attach(owner):
            await socket.send_json({"type": "error", "message": "another browser owns the control session"})
            await socket.close(code=4001)
            return
        send_lock = asyncio.Lock()
        command_tasks: set[asyncio.Task] = set()

        async def send(payload):
            async with send_lock:
                await socket.send_json(payload)

        async def telemetry_loop():
            while True:
                await send({"type": "telemetry", "state": await asyncio.to_thread(service.telemetry)})
                await asyncio.sleep(0.1)

        async def run_command(function, *args):
            try:
                await asyncio.to_thread(function, *args)
                await send({"type": "accepted"})
            except Exception as error:
                await send({"type": "error", "message": str(error)})

        sender = asyncio.create_task(telemetry_loop())
        try:
            while True:
                payload = await socket.receive_json()
                message_type = payload.get("type")
                if message_type == "stop":
                    await asyncio.to_thread(service.stop_motion, owner)
                    continue
                if message_type == "release":
                    await asyncio.to_thread(service.release_control, owner)
                    continue
                try:
                    if message_type == "jog":
                        message = JogMessage(**payload)
                        arguments = (owner, message.side.value, message.mode, message.axis,
                                     message.delta, message.duration_s, message.elbow.value)
                        task = asyncio.create_task(run_command(service.command_jog, *arguments))
                    elif message_type == "pose":
                        message = PoseMessage(**payload)
                        target = PoseTarget(message.side, message.xyz, message.rpy_deg,
                                            message.duration_s, message.elbow)
                        task = asyncio.create_task(run_command(service.command_pose, owner, target))
                    elif message_type == "aruco_pose":
                        message = ArucoPoseMessage(**payload)
                        offset = None if message.offset is None else MarkerOffset(
                            message.offset.xyz_m, message.offset.rpy_deg
                        )
                        arguments = (
                            owner,
                            message.side.value,
                            message.marker_id,
                            offset,
                            message.duration_s,
                            message.elbow.value,
                        )
                        task = asyncio.create_task(
                            run_command(service.command_aruco, *arguments)
                        )
                    elif message_type == "gate":
                        message = GateMessage(**payload)
                        offset = None if message.offset is None else MarkerOffset(
                            message.offset.xyz_m, message.offset.rpy_deg
                        )
                        arguments = (
                            owner,
                            message.side.value,
                            message.gate,
                            offset,
                            message.duration_s,
                            message.elbow.value,
                        )
                        task = asyncio.create_task(
                            run_command(service.command_gate, *arguments)
                        )
                    elif message_type == "normal":
                        duration = float(payload.get("duration_s", 20.0))
                        task = asyncio.create_task(run_command(service.command_normal, owner, duration))
                    elif message_type == "hand":
                        message = HandMessage(**payload)
                        task = asyncio.create_task(run_command(service.command_hand, owner, message.targets, message.duration_s))
                    elif message_type == "grasp":
                        message = GraspMessage(**payload)
                        task = asyncio.create_task(run_command(service.command_grasp, owner, message.name,
                                                               message.sides, message.duration_s))
                    else:
                        await send({"type": "error", "message": f"unknown message type: {message_type}"})
                        continue
                except Exception as error:
                    await send({"type": "error", "message": str(error)})
                    continue
                command_tasks.add(task)
                task.add_done_callback(command_tasks.discard)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            for task in command_tasks:
                task.cancel()
            await asyncio.to_thread(service.detach, owner)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the local Classic Control browser UI.")
    parser.add_argument("--port", type=int, default=8000, help="localhost port (default: 8000)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
