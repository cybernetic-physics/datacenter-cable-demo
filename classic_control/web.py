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
from .camera import CameraBusy, CameraUnavailable, HeadCamera
from .config import grasp_file, xr_teleoperate_root
from .grasps import DEX3_LIMITS, JOINT_NAMES, Grasp, GraspStore
from .hardware import UnitreeRobotBackend
from .models import ArmSide, ElbowBias, PoseTarget
from .service import ControlService

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


def _real_service() -> ControlService:
    root = xr_teleoperate_root()
    return ControlService(UnitreeRobotBackend(root), ArmPlanner(root), GraspStore(grasp_file()))


def create_app(control: ControlService | None = None, camera: HeadCamera | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.control = control or _real_service()
        app.state.camera = camera or HeadCamera()
        await asyncio.to_thread(app.state.control.start)
        try:
            yield
        finally:
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

    @app.get("/api/camera")
    async def camera_status():
        return await asyncio.to_thread(app.state.camera.status)

    @app.get("/api/camera.mjpg")
    def camera_stream():
        try:
            stream = app.state.camera.stream()
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
