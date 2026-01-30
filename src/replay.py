"""Replay system for playing back recorded JSON sessions through the engine."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pygame
import torch
from world_engine import CtrlInput, WorldEngine

from recorder import (
    RECORDING_FPS,
    Recorder,
    Recording,
    RecordingSettings,
    decode_video_frames,
)
from rendering import draw
from seed_gen import ENGINE_RESOLUTION, pil_to_tensor
from state import ClientState

_PROJECT_ROOT = Path(__file__).parent.parent


async def replay_from_json(
    json_path: Path,
    engine: WorldEngine,
    _screen: pygame.Surface,
    state: ClientState,
    *,
    record: bool = False,
    model_name: str = "",
    vae_uri: str = "",
    prime_seconds: float = 0.0,
) -> None:
    """Replay a recorded JSON session through the engine.

    Loads the seed image, feeds recorded controls frame-by-frame, and applies
    any injections that were captured during the original session.

    If *record* is True, a new recording is created using *model_name* /
    *vae_uri* (the currently-loaded engine) while replaying the original
    controls, producing a side-by-side comparison recording.
    """
    recording = Recording.model_validate_json(json_path.read_text())

    if recording.version != 1:
        print(f"Unsupported recording version: {recording.version}")
        return

    # Resolve seed image relative to project root
    seed_path = _PROJECT_ROOT / recording.seed_image

    from PIL import Image

    seed_pil = Image.open(seed_path)
    seed_tensor = pil_to_tensor(seed_pil, (ENGINE_RESOLUTION[1], ENGINE_RESOLUTION[0]))

    await asyncio.to_thread(engine.reset)
    _ = await asyncio.to_thread(engine.append_frame, seed_tensor)

    # Optionally start a recorder for the re-recorded output
    recorder: Recorder | None = None
    if record:
        recorder = Recorder(
            model_name=model_name,
            vae_uri=vae_uri,
            seed_frame=seed_tensor,
            initial_prompt=recording.initial_prompt,
            settings=RecordingSettings(
                n_frames=recording.settings.n_frames,
                i2i_interval=recording.settings.i2i_interval,
                i2i_vlm_regen=recording.settings.i2i_vlm_regen,
                mouse_sensitivity=recording.settings.mouse_sensitivity,
                denoise=recording.settings.denoise,
                blend_falloff=recording.settings.blend_falloff,
                click_repainting=recording.settings.click_repainting,
            ),
        )

    # Sort injections by after_frame
    sorted_injections = sorted(recording.injections, key=lambda inj: inj.after_frame)
    injection_idx = 0

    # Decode original frames for primed re-recording
    prime_frame_count = 0
    original_frames: list[torch.Tensor] = []
    if prime_seconds > 0 and record:
        mp4_path = json_path.with_suffix(".mp4")
        prime_frame_count = min(
            int(prime_seconds * RECORDING_FPS), len(recording.frames)
        )
        w, h = ENGINE_RESOLUTION
        original_frames = decode_video_frames(mp4_path, prime_frame_count, w, h)
        # Clamp to actually decoded count
        prime_frame_count = len(original_frames)

    try:
        for frame_idx, frame_rec in enumerate(recording.frames):
            # Check for cancellation via pygame events
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:  # pyright: ignore[reportAny]
                    return

            mouse = frame_rec.ctrl.mouse
            ctrl_input = CtrlInput(
                button=set(frame_rec.ctrl.button),
                mouse=(mouse[0], mouse[1]),
                scroll_wheel=frame_rec.ctrl.scroll_wheel,
            )

            if frame_idx < prime_frame_count:
                # Phase 1: feed original frame + controls into the engine context
                img = original_frames[frame_idx]
                _ = await asyncio.to_thread(engine.append_frame, img, ctrl=ctrl_input)
            else:
                # Phase 2: generate with the (possibly new) model
                img = await asyncio.to_thread(  # pyright: ignore[reportUnknownVariableType]
                    engine.gen_frame, ctrl=ctrl_input
                )

            draw(img, state)  # pyright: ignore[reportUnknownArgumentType]

            if recorder is not None:
                recorder.record_frame(frame_rec.index, ctrl_input, img)  # pyright: ignore[reportUnknownArgumentType]

            # Check for pending injections at this frame index
            while injection_idx < len(sorted_injections):
                inj = sorted_injections[injection_idx]
                if inj.after_frame > frame_rec.index:
                    break
                # Load injection image and append to engine
                inj_path = _PROJECT_ROOT / inj.image
                inj_pil = Image.open(inj_path)
                inj_tensor = pil_to_tensor(
                    inj_pil, (ENGINE_RESOLUTION[1], ENGINE_RESOLUTION[0])
                )
                _ = await asyncio.to_thread(engine.append_frame, inj_tensor)
                if recorder is not None:
                    recorder.record_injection(
                        inj.after_frame, inj_tensor, inj.prompt, inj.type
                    )
                injection_idx += 1

            await asyncio.sleep(0)
    finally:
        if recorder is not None:
            path = recorder.finalize()
            print(f"Re-recorded to {path}")
