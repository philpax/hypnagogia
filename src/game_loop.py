"""Main game loop for the client."""

import asyncio
import time
from pathlib import Path

import pygame
import torch
from world_engine import CtrlInput

from blending import blend_frames, create_blend_mask
from config import get_config, load_user_config
from engine import SLEEP_RATIO, Engine
from seed_gen import ENGINE_RESOLUTION, generate_i2i, generate_t2i
from vision_api import VisionResult, describe_frame

from constants import (
    HISTORY_BROWSE_KEY,
    ImageHistoryEntry,
    i2i_executor,
    vision_executor,
)
from input import ctrl_stream
from pause_menu import show_pause_menu
from recorder import RERECORD_PRIME_SECONDS, Recorder, RecordingSettings
from rendering import render_batch
from replay import replay_from_json
from state import ClientState, GameState


async def run_loop(
    *,
    engine: Engine,
    seed_frame: torch.Tensor | None,
    n_frames: int,
    mouse_sensitivity: float,
    comfyui_url: str | None = None,
    prompt: str | None = None,
    image_seed: int | None = None,
    i2i_interval: int,
    i2i_vlm_regen: bool,
    denoise: float,
    vision_api_url: str,
    vision_model: str,
) -> None:
    """Main game loop that handles rendering, input, and state management."""
    config = get_config()
    _ = pygame.init()
    _ = pygame.mixer.init()
    screen = pygame.display.set_mode(
        (config.window.width, config.window.height), pygame.RESIZABLE
    )
    pygame.scrap.init()  # Enable clipboard support (must be after display init)
    pygame.display.set_caption(
        f"hypnagogia (esc to pause, u to reset, {pygame.key.name(HISTORY_BROWSE_KEY)} to browse history)"
    )

    # Load sound effects
    sounds_dir = Path(__file__).parent.parent / "sounds"
    lmb_sound = pygame.mixer.Sound(sounds_dir / "foom_0.wav")
    rmb_sound = pygame.mixer.Sound(sounds_dir / "alert-beep.mp3")

    # Load user config
    user_config = load_user_config()

    # Initialize client state
    state = ClientState(
        screen=screen,
        lmb_sound=lmb_sound,
        rmb_sound=rmb_sound,
        seed_frame=seed_frame,
        prompt=prompt,
        current_denoise=denoise,
        show_history_previews=user_config.show_history_previews,
        show_prompt=user_config.show_prompt,
        blend_falloff=user_config.blend_falloff,
        click_repainting=user_config.click_repainting,
        recording_enabled=user_config.recording_enabled,
    )

    tc = engine.temporal_compression
    fps_cap = engine.inference_fps

    try:
        # Apply initial game state (PAUSED - cursor visible, not grabbed)
        state.apply_game_state()

        restart = asyncio.Event()
        pause = asyncio.Event()
        limit = max(1, n_frames - 2)

        # Pipeline state: the most recent CPU batch waiting to be rendered,
        # and the ctrl that produced it (so the recorder logs the right ctrl).
        pending: torch.Tensor | None = None
        pending_ctrl: CtrlInput = CtrlInput()
        batch_dt = 0.0
        overhead = 0.0
        pace_s = 0.0

        async def reset(*, reload_seed: bool = False) -> None:
            nonlocal pending, pending_ctrl
            # Finalize any active recorder before resetting
            if state.recorder is not None:
                _ = state.recorder.finalize()
                state.recorder = None

            state.play_time = 0.0
            if state.play_start is not None:
                state.play_start = (
                    time.time()
                )  # Restart play timer if currently playing
            await asyncio.to_thread(engine.reset)
            if reload_seed or state.seed_frame is None:
                if comfyui_url and state.prompt:
                    state.seed_frame = await asyncio.to_thread(
                        generate_t2i, comfyui_url, state.prompt, image_seed
                    )
                else:
                    raise ValueError(
                        "ComfyUI URL and prompt are required for seed generation"
                    )
            # Always reset history to just the seed frame
            state.image_history = []
            state.history_cache.clear()
            state.history_scroll = 0
            # seed_frame is guaranteed non-None here (assigned above or was already set)
            if state.prompt is not None:
                state.image_history.insert(
                    0, ImageHistoryEntry(image=state.seed_frame, prompt=state.prompt)
                )
                _ = await asyncio.to_thread(engine.append_frame, state.seed_frame)

            # Prime the pipeline: produce one initial CPU batch so the loop
            # has something to render on its first iteration. The first call
            # also triggers torch.compile for WP-1.5; subsequent resets are
            # fast.
            pending = await asyncio.to_thread(engine.warmup)
            pending_ctrl = CtrlInput()

            # Start new recorder if recording is enabled and we have a real session
            if state.recording_enabled and initial_pause_done:
                assert state.seed_frame is not None
                state.recorder = Recorder(
                    model_name=config.models.world_engine,
                    vae_uri=config.models.vae_uri,
                    seed_frame=state.seed_frame,
                    initial_prompt=state.prompt or "",
                    settings=RecordingSettings(
                        n_frames=n_frames,
                        i2i_interval=i2i_interval,
                        i2i_vlm_regen=i2i_vlm_regen,
                        mouse_sensitivity=mouse_sensitivity,
                        denoise=state.current_denoise,
                        blend_falloff=state.blend_falloff,
                        click_repainting=state.click_repainting,
                    ),
                    fps=fps_cap,
                )

        def handle_scroll(y: int) -> None:
            # Check if mouse is over history area (top-left)
            mouse_x, _ = pygame.mouse.get_pos()
            sw, _ = screen.get_size()
            history_width = sw // 8  # thumb_width, no padding
            if mouse_x < history_width and state.image_history:
                # Scroll history (y > 0 = scroll up = decrease offset)
                max_scroll = max(0, len(state.image_history) - 1)
                state.history_scroll = max(0, min(max_scroll, state.history_scroll - y))
            else:
                # y > 0 means scroll up (increase), y < 0 means scroll down (decrease)
                state.current_denoise = max(
                    0.0, min(1.0, state.current_denoise + y * 0.05)
                )

        def handle_history_click(pos: tuple[int, int]) -> None:
            """Handle clicking on a history item to set its prompt."""
            if not state.image_history:
                return

            mouse_x, mouse_y = pos
            sw, sh = screen.get_size()
            thumb_width = sw // 8
            thumb_height = sh // 8
            estimated_prompt_height = 30
            entry_height = thumb_height + estimated_prompt_height

            # Check if click is within history area
            if mouse_x > thumb_width:
                return

            # Calculate which entry was clicked (no offset, starts at y=0)
            entry_index = state.history_scroll + (mouse_y // entry_height)
            if 0 <= entry_index < len(state.image_history):
                entry = state.image_history[entry_index]
                state.prompt = entry.prompt
                state.invalidate_prompt_cache()
                print(f"Set prompt from history: {state.prompt}")

        ctrls = ctrl_stream(
            restart_event=restart,
            pause_event=pause,
            mouse_sensitivity=mouse_sensitivity,
            state=state,
            on_scroll=handle_scroll,
            on_history_click=handle_history_click,
        )

        # Track blend falloff to detect changes
        last_blend_falloff = state.blend_falloff

        # Warm up engine with a black frame instead of generating via T2I
        state.seed_frame = torch.zeros(
            ENGINE_RESOLUTION[1], ENGINE_RESOLUTION[0], 3, dtype=torch.uint8
        )
        initial_pause_done = False
        await reset(reload_seed=False)

        state.frames = 0

        def _record_sub(sub: torch.Tensor, _i: int) -> None:
            """Per-sub-frame side effects: counter, recorder, last_frame."""
            state.last_frame = sub
            if state.recorder is not None:
                state.recorder.record_frame(state.frames, pending_ctrl, sub)
            state.frames += 1

        async for ctrl in ctrls:
            # Regenerate mask if falloff changed
            if state.blend_falloff != last_blend_falloff:
                state.blend_mask = None
                last_blend_falloff = state.blend_falloff

            # Check if i2i task completed (non-blocking)
            if state.i2i_future is not None and state.i2i_future.done():
                refreshed = state.i2i_future.result()
                # Blend i2i result with last world model frame using blend mask
                if state.last_frame is not None:  # pyright: ignore[reportUnknownMemberType]
                    h, w = refreshed.shape[:2]
                    if (
                        state.blend_mask is None
                        or state.blend_mask.shape[0] != h
                        or state.blend_mask.shape[1] != w
                    ):
                        state.blend_mask = create_blend_mask(h, w, state.blend_falloff)
                    refreshed = blend_frames(
                        refreshed,
                        state.last_frame,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                        state.blend_mask,
                    )
                _ = await asyncio.to_thread(engine.append_frame, refreshed)
                if state.recorder is not None and state.i2i_pending_prompt:
                    state.recorder.record_injection(
                        state.frames,
                        refreshed,
                        state.i2i_pending_prompt,
                        "i2i_append",
                    )
                # Add to image history
                if state.i2i_pending_prompt:
                    state.image_history.insert(
                        0,
                        ImageHistoryEntry(
                            image=refreshed, prompt=state.i2i_pending_prompt
                        ),
                    )
                state.i2i_future = None
                state.i2i_pending_prompt = None

            # Check if vision task completed (non-blocking)
            if state.vision_future is not None and state.vision_future.done():
                try:
                    vision_result: VisionResult = state.vision_future.result()
                    if vision_result.success:
                        state.prompt = vision_result.prompt
                        state.invalidate_prompt_cache()
                        print(f"Vision: {state.prompt}")
                    else:
                        print(f"Vision error: {vision_result.error}")
                except Exception as e:
                    print(f"Vision exception: {e}")
                state.vision_future = None

            # Check if VLM-i2i vision task completed (non-blocking)
            if (
                state.vlm_i2i_vision_future is not None
                and state.vlm_i2i_vision_future.done()
            ):
                try:
                    vlm_i2i_result: VisionResult = state.vlm_i2i_vision_future.result()
                    if vlm_i2i_result.success:
                        state.prompt = vlm_i2i_result.prompt
                        state.invalidate_prompt_cache()
                        print(f"VLM-i2i: {state.prompt}")
                        # Now trigger i2i with the new prompt and current frame
                        if (
                            state.i2i_future is None
                            and comfyui_url
                            and state.prompt
                            and state.last_frame is not None  # pyright: ignore[reportUnknownMemberType]
                        ):
                            state.i2i_pending_prompt = state.prompt
                            state.i2i_future = i2i_executor.submit(
                                generate_i2i,
                                comfyui_url,
                                state.prompt,
                                state.last_frame,
                                None,
                                state.current_denoise,
                            )
                    else:
                        print(f"VLM-i2i error: {vlm_i2i_result.error}")
                except Exception as e:
                    print(f"VLM-i2i exception: {e}")
                state.vlm_i2i_vision_future = None

            if pause.is_set():
                pause.clear()
                # Flush any pending batch so the pause menu has the latest
                # frame as its background and the recorder records it.
                if pending is not None:
                    render_batch(pending, state, 0.0, on_sub_frame=_record_sub)
                    pending = None
                # Finalize recording on pause so each recording is one
                # uninterrupted play segment (reset → pause)
                if state.recorder is not None:
                    _ = state.recorder.finalize()
                    state.recorder = None
                result = await show_pause_menu(
                    screen,
                    state,
                    state.last_frame,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                    comfyui_url,
                    image_seed,
                )

                if result.action == "quit":
                    return

                if result.action == "regenerate":
                    assert result.new_prompt is not None
                    assert result.regenerated_frame is not None
                    state.prompt = result.new_prompt  # Update prompt
                    state.current_denoise = result.denoise  # Update denoise value
                    state.invalidate_prompt_cache()

                    if result.reset_with_seed:
                        # Finalize any active recorder before resetting
                        if state.recorder is not None:
                            _ = state.recorder.finalize()
                            state.recorder = None

                        # Reset engine with new T2I seed
                        state.play_time = 0.0
                        await asyncio.to_thread(engine.reset)
                        state.seed_frame = result.regenerated_frame
                        _ = await asyncio.to_thread(
                            engine.append_frame, state.seed_frame
                        )
                        # Re-prime pipeline after reseed
                        pending = await asyncio.to_thread(engine.warmup)
                        pending_ctrl = CtrlInput()
                        state.frames = 0
                        # Clear history on T2I and add new seed
                        state.image_history.clear()
                        state.history_cache.clear()
                        state.history_scroll = 0
                        state.image_history.insert(
                            0,
                            ImageHistoryEntry(
                                image=state.seed_frame, prompt=state.prompt
                            ),
                        )

                        # Start new recorder if recording is enabled
                        if state.recording_enabled:
                            state.recorder = Recorder(
                                model_name=config.models.world_engine,
                                vae_uri=config.models.vae_uri,
                                seed_frame=state.seed_frame,
                                initial_prompt=state.prompt or "",
                                settings=RecordingSettings(
                                    n_frames=n_frames,
                                    i2i_interval=i2i_interval,
                                    i2i_vlm_regen=i2i_vlm_regen,
                                    mouse_sensitivity=mouse_sensitivity,
                                    denoise=state.current_denoise,
                                    blend_falloff=state.blend_falloff,
                                    click_repainting=state.click_repainting,
                                ),
                                fps=fps_cap,
                            )
                    else:
                        # Append I2I regenerated frame and continue
                        # Blend with last world model frame using blend mask
                        blended_frame = result.regenerated_frame
                        if state.last_frame is not None:  # pyright: ignore[reportUnknownMemberType]
                            h, w = blended_frame.shape[:2]
                            if (
                                state.blend_mask is None
                                or state.blend_mask.shape[0] != h
                                or state.blend_mask.shape[1] != w
                            ):
                                state.blend_mask = create_blend_mask(
                                    h, w, state.blend_falloff
                                )
                            blended_frame = blend_frames(
                                blended_frame,
                                state.last_frame,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                                state.blend_mask,
                            )
                        _ = await asyncio.to_thread(engine.append_frame, blended_frame)
                        if state.recorder is not None:
                            state.recorder.record_injection(
                                state.frames,
                                blended_frame,
                                state.prompt or "",
                                "pause_i2i",
                            )
                        state.last_frame = blended_frame
                        # Add to image history
                        state.image_history.insert(
                            0,
                            ImageHistoryEntry(image=blended_frame, prompt=state.prompt),
                        )

                # Detect recording toggled off from pause menu
                if not state.recording_enabled and state.recorder is not None:
                    _ = state.recorder.finalize()
                    state.recorder = None

                if result.action == "replay":
                    assert result.replay_json_path is not None
                    await replay_from_json(
                        result.replay_json_path, engine, screen, state
                    )
                    pause.set()
                    continue

                if result.action == "rerecord":
                    assert result.replay_json_path is not None
                    await replay_from_json(
                        result.replay_json_path,
                        engine,
                        screen,
                        state,
                        record=True,
                        model_name=config.models.world_engine,
                        vae_uri=config.models.vae_uri,
                    )
                    pause.set()
                    continue

                if result.action == "rerecord_primed":
                    assert result.replay_json_path is not None
                    await replay_from_json(
                        result.replay_json_path,
                        engine,
                        screen,
                        state,
                        record=True,
                        model_name=config.models.world_engine,
                        vae_uri=config.models.vae_uri,
                        prime_seconds=RERECORD_PRIME_SECONDS,
                    )
                    pause.set()
                    continue

                # Start play timer when resuming
                state.play_start = time.time()
                state.game_state = GameState.PLAYING
                state.apply_game_state()
                _ = pygame.mouse.get_rel()  # discard accumulated mouse movement
                # Reset button state to avoid triggering I2I/vision from UI clicks
                state.lmb_was_pressed = True
                state.rmb_was_pressed = True
                continue

            if restart.is_set() or state.frames >= limit:
                restart.clear()
                await reset(reload_seed=False)
                state.frames = 0

            if state.click_repainting:
                # RMB edge detection for vision API (before gen_frame)
                rmb_pressed = 0x02 in ctrl.button
                if (
                    rmb_pressed
                    and not state.rmb_was_pressed
                    and state.vision_future is None
                    and state.last_frame is not None  # pyright: ignore[reportUnknownMemberType]
                ):
                    _ = state.rmb_sound.play()
                    state.vision_future = vision_executor.submit(
                        describe_frame,
                        state.last_frame.clone(),
                        vision_api_url,
                        vision_model,
                        config.vision.api_key_env,
                        config.vision.max_tokens,
                        config.vision.timeout,
                    )
                state.rmb_was_pressed = rmb_pressed

                # LMB edge detection for i2i submission (before gen_frame)
                lmb_pressed = 0x01 in ctrl.button
                if (
                    lmb_pressed
                    and not state.lmb_was_pressed
                    and state.i2i_future is None
                    and comfyui_url
                    and state.prompt
                    and state.last_frame is not None  # pyright: ignore[reportUnknownMemberType]
                ):
                    _ = state.lmb_sound.play()
                    state.i2i_pending_prompt = state.prompt  # Track prompt for history
                    state.i2i_future = i2i_executor.submit(
                        generate_i2i,
                        comfyui_url,
                        state.prompt,
                        state.last_frame,
                        None,
                        state.current_denoise,
                    )
                state.lmb_was_pressed = lmb_pressed

                # Filter out LMB and RMB from ctrl before sending to world model
                filtered_buttons = ctrl.button - {0x01, 0x02}
                filtered_ctrl = CtrlInput(button=filtered_buttons, mouse=ctrl.mouse)
            else:
                # Pass clicks through to the world model
                filtered_ctrl = ctrl

            # --- Frame pacing pipeline (mirrors world_engine examples/interactive.py)
            # 1. Dispatch gen_frame — GPU kernels are queued, returns fast.
            # 2. Render the *previous* batch with pacing sleeps while GPU works.
            # 3. .cpu() syncs the GPU and transfers the just-computed batch.
            # 4. Measure overhead (non-render time) to feed back into pacing.
            t0 = time.perf_counter()
            next_frames_gpu: torch.Tensor = await asyncio.to_thread(  # pyright: ignore[reportUnknownVariableType]
                engine.gen_frame, ctrl=filtered_ctrl
            )

            if pending is not None:
                # Target visual interval for this batch (T sub-frames at fps_cap).
                # Subtract measured overhead so the *total* cycle hits target_s;
                # floor at SLEEP_RATIO*batch_dt to prevent render-time feedback
                # loops when the GPU is slower than the cap.
                target_s = tc / fps_cap if fps_cap > 0 else 0.0
                pace_s = max(batch_dt * SLEEP_RATIO, target_s - overhead)
                render_batch(pending, state, pace_s, on_sub_frame=_record_sub)

            pending = await asyncio.to_thread(next_frames_gpu.cpu)  # pyright: ignore[reportUnknownArgumentType]
            pending_ctrl = filtered_ctrl
            batch_dt = time.perf_counter() - t0
            overhead = batch_dt - pace_s

            # Pause after the first rendered batch so engine is warm before
            # the pause menu appears.
            if not initial_pause_done and state.frames > 0:
                initial_pause_done = True
                pause.set()

            # Start i2i regeneration every i2i_interval sub-frames (if not
            # already running). Cross-boundary check is robust to tc>1, where
            # state.frames advances by tc per batch and a plain modulus would
            # skip triggers that fall mid-batch.
            prev_frames = max(0, state.frames - tc)
            if (
                i2i_interval > 0
                and state.frames > 0
                and state.frames // i2i_interval > prev_frames // i2i_interval
                and comfyui_url
                and state.prompt
            ):
                if i2i_vlm_regen:
                    # VLM-triggered: first get new prompt from VLM, then i2i
                    if (
                        state.vlm_i2i_vision_future is None
                        and state.last_frame is not None  # pyright: ignore[reportUnknownMemberType]
                    ):
                        state.vlm_i2i_vision_future = vision_executor.submit(
                            describe_frame,
                            state.last_frame.clone(),
                            vision_api_url,
                            vision_model,
                            config.vision.api_key_env,
                            config.vision.max_tokens,
                            config.vision.timeout,
                        )
                elif state.i2i_future is None:
                    # Direct i2i without VLM
                    state.i2i_pending_prompt = state.prompt  # Track prompt for history
                    state.i2i_future = i2i_executor.submit(
                        generate_i2i,
                        comfyui_url,
                        state.prompt,
                        state.last_frame,
                        None,
                        state.current_denoise,
                    )

            await asyncio.sleep(0)
    finally:
        if state.recorder is not None:
            _ = state.recorder.finalize()
            state.recorder = None
        pygame.event.set_grab(False)
        pygame.mixer.quit()
        pygame.quit()
