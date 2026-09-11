"""Small GStreamer bridge from the G1 head camera to browser MJPEG."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path


class CameraUnavailable(RuntimeError):
    """Raised when the configured head camera cannot be opened."""


class CameraBusy(RuntimeError):
    """Raised when another browser is already consuming the camera."""


class HeadCamera:
    """Expose the AIRHUG UVC camera without decoding its native JPEG frames."""

    def __init__(self, device: str | None = None):
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

        for entry in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                if entry.joinpath("name").read_text().strip() == "AIRHUG 02: AIRHUG 02":
                    if entry.joinpath("index").read_text().strip() == "0":
                        return Path("/dev") / entry.name
            except OSError:
                continue
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
            error = "AIRHUG 02 head camera was not found"
        return {
            "available": bool(gst and device is not None and device.exists()),
            "streaming": streaming,
            "device": str(device) if device is not None else None,
            "resolution": [1280, 720],
            "error": error,
        }

    def stream(self) -> Iterator[bytes]:
        if not self._stream_lock.acquire(blocking=False):
            raise CameraBusy("the head camera is already streaming")

        status = self.status()
        if not status["available"]:
            self._stream_lock.release()
            raise CameraUnavailable(str(status["error"]))

        command = [
            "gst-launch-1.0", "-q",
            "v4l2src", f"device={status['device']}",
            "!", "image/jpeg,width=1280,height=720",
            "!", "multipartmux", "boundary=frame",
            "!", "fdsink", "fd=1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, ValueError) as error:
            self._stream_lock.release()
            raise CameraUnavailable(f"could not start the head camera: {error}") from error

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
                detail = ""
                if process.stderr is not None:
                    detail = process.stderr.read().decode(errors="replace").strip().splitlines()[-1:]
                    detail = detail[0] if detail else ""
                with self._state_lock:
                    self._last_error = detail or f"camera stream exited with status {return_code}"
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
