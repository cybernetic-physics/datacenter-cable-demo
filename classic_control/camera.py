"""Camera sources exposed as browser-compatible MJPEG streams."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import zmq


class CameraUnavailable(RuntimeError):
    """Raised when a camera source cannot be opened."""


class CameraBusy(RuntimeError):
    """Raised when another browser is already consuming a camera source."""


def multipart_jpeg(jpeg: bytes) -> bytes:
    """Wrap one JPEG for an HTTP multipart MJPEG response."""
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(jpeg)).encode()
        + b"\r\n\r\n"
        + jpeg
        + b"\r\n"
    )


class TeleimagerCamera:
    """Turn Teleimager's latest-value JPEG ZMQ stream into multipart MJPEG."""

    def __init__(self, name: str, host: str, port: int, resolution: tuple[int, int]):
        self.name = name
        self.host = host
        self.port = port
        self.resolution = resolution
        self._stream_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._streaming = False
        self._last_error: str | None = None

    def status(self) -> dict[str, object]:
        try:
            connection = socket.create_connection((self.host, self.port), timeout=0.25)
            connection.close()
            available = True
            error = None
        except OSError:
            available = False
            error = f"Teleimager is not reachable at {self.host}:{self.port}"
        with self._state_lock:
            streaming = self._streaming
            if self._last_error and not available:
                error = self._last_error
        return {
            "name": self.name,
            "available": available,
            "streaming": streaming,
            "location": f"{self.host}:{self.port}",
            "resolution": list(self.resolution),
            "error": error,
        }

    def stream(self) -> Iterator[bytes]:
        frames = self.jpeg_stream()
        return self._multipart_stream(frames)

    def jpeg_stream(self) -> Iterator[bytes]:
        """Return validated JPEG frames for browser or vision consumers."""
        if not self._stream_lock.acquire(blocking=False):
            raise CameraBusy(f"{self.name} is already streaming")
        if not self.status()["available"]:
            self._stream_lock.release()
            raise CameraUnavailable(f"Teleimager is not reachable at {self.host}:{self.port}")

        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.CONFLATE, 1)
        subscriber.setsockopt(zmq.RCVHWM, 1)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.RCVTIMEO, 250)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect(f"tcp://{self.host}:{self.port}")
        self._stop_event.clear()
        with self._state_lock:
            self._streaming = True
            self._last_error = None
        return self._read_jpegs(context, subscriber)

    def _read_jpegs(self, context: zmq.Context, subscriber: zmq.Socket) -> Iterator[bytes]:
        last_frame = time.monotonic()
        try:
            while not self._stop_event.is_set():
                try:
                    jpeg = subscriber.recv()
                except zmq.Again:
                    if time.monotonic() - last_frame < 3.0:
                        continue
                    with self._state_lock:
                        self._last_error = f"No frames received from {self.host}:{self.port}"
                    break
                if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                    continue
                last_frame = time.monotonic()
                yield jpeg
        finally:
            subscriber.close(linger=0)
            context.term()
            with self._state_lock:
                self._streaming = False
            self._stream_lock.release()

    @staticmethod
    def _multipart_stream(frames: Iterator[bytes]) -> Iterator[bytes]:
        try:
            for jpeg in frames:
                yield multipart_jpeg(jpeg)
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()

    def close(self) -> None:
        self._stop_event.set()


class LocalMjpegCamera:
    """Optional Thor-attached UVC fallback using native JPEG frames."""

    def __init__(self, name: str = "Thor AIRHUG fallback", device: str | None = None):
        self.name = name
        self.resolution = (1280, 720)
        self._configured_device = device or os.environ.get("CLASSIC_CONTROL_CAMERA_DEVICE")
        self._stream_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._last_error: str | None = None

    def _device(self) -> Path | None:
        if self._configured_device:
            return Path(self._configured_device)
        by_id = Path("/dev/v4l/by-id")
        if by_id.is_dir():
            candidates = sorted(by_id.glob("*AIRHUG_02*-video-index0"))
            if candidates:
                return candidates[0]
        return None

    def status(self) -> dict[str, object]:
        device = self._device()
        gst = shutil.which("gst-launch-1.0")
        with self._state_lock:
            streaming = self._process is not None and self._process.poll() is None
            error = self._last_error
        if not gst:
            error = "gst-launch-1.0 is not installed"
        elif device is None or not device.exists():
            error = "Thor AIRHUG camera was not found"
        return {
            "name": self.name,
            "available": bool(gst and device is not None and device.exists()),
            "streaming": streaming,
            "location": str(device) if device is not None else None,
            "resolution": list(self.resolution),
            "error": error,
        }

    def stream(self) -> Iterator[bytes]:
        if not self._stream_lock.acquire(blocking=False):
            raise CameraBusy(f"{self.name} is already streaming")
        status = self.status()
        if not status["available"]:
            self._stream_lock.release()
            raise CameraUnavailable(str(status["error"]))
        command = [
            "gst-launch-1.0", "-q", "v4l2src", f"device={status['location']}",
            "!", "image/jpeg,width=1280,height=720",
            "!", "multipartmux", "boundary=frame", "!", "fdsink", "fd=1",
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (OSError, ValueError) as error:
            self._stream_lock.release()
            raise CameraUnavailable(f"could not start Thor camera: {error}") from error
        with self._state_lock:
            self._process = process
            self._last_error = None
        return self._read_stream(process)

    def _read_stream(self, process: subprocess.Popen[bytes]) -> Iterator[bytes]:
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read1(64 * 1024):
                yield chunk
            return_code = process.wait()
            if return_code != 0:
                with self._state_lock:
                    self._last_error = f"Thor camera exited with status {return_code}"
        finally:
            self._stop_process(process)
            self._stream_lock.release()

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        with self._state_lock:
            if self._process is process:
                self._process = None

    def close(self) -> None:
        with self._state_lock:
            process = self._process
        if process is not None:
            self._stop_process(process)


class CameraHub:
    """Named camera sources kept independent from robot control state."""

    def __init__(self, sources: dict[str, object] | None = None):
        host = os.environ.get("TELEIMAGER_HOST", "192.168.123.164")
        self.sources = sources or {
            "internal": TeleimagerCamera("Internal D435i RGB", host, 55555, (640, 480)),
            "stereo": TeleimagerCamera("Taped head stereo", host, 55556, (1280, 480)),
            "thor": LocalMjpegCamera(),
        }

    def status(self) -> dict[str, object]:
        return {
            "default": "stereo",
            "sources": {source_id: source.status() for source_id, source in self.sources.items()},
        }

    def stream(self, source_id: str) -> Iterator[bytes]:
        try:
            source = self.sources[source_id]
        except KeyError as error:
            raise KeyError(f"unknown camera source: {source_id}") from error
        return source.stream()

    def jpeg_stream(self, source_id: str) -> Iterator[bytes]:
        try:
            source = self.sources[source_id]
        except KeyError as error:
            raise KeyError(f"unknown camera source: {source_id}") from error
        try:
            return source.jpeg_stream()
        except AttributeError as error:
            raise CameraUnavailable(f"{source_id} does not expose JPEG frames") from error

    def teleimager_config(self) -> dict[str, object]:
        """Read the active PC2 camera profile from Teleimager's responder."""
        source = self.sources.get("internal")
        if not isinstance(source, TeleimagerCamera):
            raise CameraUnavailable("the internal camera is not a Teleimager source")
        context = zmq.Context()
        requester = context.socket(zmq.REQ)
        requester.setsockopt(zmq.LINGER, 0)
        requester.setsockopt(zmq.RCVTIMEO, 1000)
        requester.setsockopt(zmq.SNDTIMEO, 1000)
        requester.connect(f"tcp://{source.host}:60000")
        try:
            requester.send(b"GET_DATA")
            config = requester.recv_json()
            if not isinstance(config, dict):
                raise CameraUnavailable("Teleimager returned an invalid camera configuration")
            return config
        except zmq.ZMQError as error:
            raise CameraUnavailable(f"Teleimager camera configuration failed: {error}") from error
        finally:
            requester.close(linger=0)
            context.term()

    def close(self) -> None:
        for source in self.sources.values():
            source.close()
