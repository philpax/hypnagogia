"""Recording system for capturing game sessions to MP4 + JSON."""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from pydantic import BaseModel

from seed_gen import ENGINE_RESOLUTION

_PROJECT_ROOT = Path(__file__).parent.parent
_RECORDINGS_DIR = _PROJECT_ROOT / "recordings"

# Default playback FPS for legacy recordings made before fps was persisted.
# Live recordings use the engine's ``inference_fps`` (passed to ``Recorder``).
RECORDING_FPS: float = 30.0
RERECORD_PRIME_SECONDS: float = 2.0


# ── Pydantic models for the recording JSON ──────────────────────────────


class CtrlRecord(BaseModel):
    button: list[int]
    mouse: list[float]
    scroll_wheel: int = 0


class FrameRecord(BaseModel):
    index: int
    ctrl: CtrlRecord


class InjectionRecord(BaseModel):
    after_frame: int
    type: str
    image: str
    prompt: str


class RecordingSettings(BaseModel):
    n_frames: int
    i2i_interval: int
    i2i_vlm_regen: bool
    mouse_sensitivity: float
    denoise: float
    blend_falloff: float
    click_repainting: bool


class Recording(BaseModel):
    version: int = 1
    timestamp: str
    model: str
    vae_uri: str | None = None
    seed_image: str
    initial_prompt: str
    settings: RecordingSettings
    frames: list[FrameRecord]
    injections: list[InjectionRecord]
    fps: float = RECORDING_FPS


# ── VideoWriter ─────────────────────────────────────────────────────────


class VideoWriter:
    """Pipes raw RGB24 frames to an ffmpeg subprocess (H.264, CRF 18, yuv420p)."""

    _proc: subprocess.Popen[bytes]

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found on PATH; install it to enable recording"
            )
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_frame(self, frame_bytes: bytes) -> None:
        assert self._proc.stdin is not None
        _ = self._proc.stdin.write(frame_bytes)

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        _ = self._proc.wait()


def decode_video_frames(
    mp4_path: Path, n_frames: int, width: int, height: int
) -> list[torch.Tensor]:
    """Decode up to *n_frames* RGB24 frames from *mp4_path* via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; install it to enable video decoding"
        )
    proc = subprocess.Popen(
        [
            ffmpeg,
            "-i",
            str(mp4_path),
            "-frames:v",
            str(n_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    frame_size = width * height * 3
    frames: list[torch.Tensor] = []
    assert proc.stdout is not None
    for _ in range(n_frames):
        data = proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(
            height, width, 3
        )
        frames.append(tensor)
    proc.stdout.close()
    _ = proc.wait()
    return frames


# ── Recorder ────────────────────────────────────────────────────────────


def _sanitize_model_name(model: str) -> str:
    """Make a model name safe for use in filenames on Windows."""
    # Take just the last component if it looks like a path
    name = model.rsplit("/", 1)[-1]
    # Replace Windows-unsafe characters
    return re.sub(r'[\\/:*?"<>|]', "_", name)


class Recorder:
    """Manages one recording session (MP4 + JSON + seed PNG + injection PNGs)."""

    _stem: str
    _video: VideoWriter
    _timestamp: str
    _model_name: str
    _vae_uri: str | None
    _seed_image_rel: str
    _initial_prompt: str
    _settings: RecordingSettings

    def __init__(
        self,
        model_name: str,
        vae_uri: str | None,
        seed_frame: torch.Tensor,
        initial_prompt: str,
        settings: RecordingSettings,
        fps: float = RECORDING_FPS,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_model = _sanitize_model_name(model_name)
        self._stem = f"{ts}-{safe_model}"
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Save seed frame as PNG
        seed_path = _RECORDINGS_DIR / f"{self._stem}_seed.png"
        arr = seed_frame.cpu().numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        Image.fromarray(arr).save(seed_path)  # pyright: ignore[reportUnknownArgumentType]

        # Start video writer
        w, h = ENGINE_RESOLUTION
        self._fps = fps
        self._video = VideoWriter(_RECORDINGS_DIR / f"{self._stem}.mp4", w, h, fps)

        self._timestamp = ts
        self._model_name = model_name
        self._vae_uri = vae_uri
        self._seed_image_rel = f"recordings/{self._stem}_seed.png"
        self._initial_prompt = initial_prompt
        self._settings = settings
        self._frames: list[FrameRecord] = []
        self._injections: list[InjectionRecord] = []

    def record_frame(
        self, frame_index: int, ctrl: object, frame_tensor: torch.Tensor
    ) -> None:
        from world_engine import CtrlInput

        assert isinstance(ctrl, CtrlInput)
        self._frames.append(
            FrameRecord(
                index=frame_index,
                ctrl=CtrlRecord(
                    button=sorted(ctrl.button),
                    mouse=list(ctrl.mouse),
                    scroll_wheel=ctrl.scroll_wheel,
                ),
            )
        )
        raw = frame_tensor.cpu().numpy().tobytes()  # pyright: ignore[reportUnknownMemberType]
        self._video.write_frame(raw)

    def record_injection(
        self,
        frame_index: int,
        injected_frame: torch.Tensor,
        prompt: str,
        injection_type: str,
    ) -> None:
        inject_name = f"{self._stem}_inject_{frame_index:04d}.png"
        inject_path = _RECORDINGS_DIR / inject_name
        arr = injected_frame.cpu().numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        Image.fromarray(arr).save(inject_path)  # pyright: ignore[reportUnknownArgumentType]
        self._injections.append(
            InjectionRecord(
                after_frame=frame_index,
                type=injection_type,
                image=f"recordings/{inject_name}",
                prompt=prompt,
            )
        )

    def finalize(self) -> Path:
        self._video.close()
        recording = Recording(
            timestamp=self._timestamp,
            model=self._model_name,
            vae_uri=self._vae_uri,
            seed_image=self._seed_image_rel,
            initial_prompt=self._initial_prompt,
            settings=self._settings,
            frames=self._frames,
            injections=self._injections,
            fps=self._fps,
        )
        json_path = _RECORDINGS_DIR / f"{self._stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            _ = f.write(recording.model_dump_json(indent=2))
            _ = f.write("\n")
        return json_path
