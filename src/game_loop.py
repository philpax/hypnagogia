"""Main game loop for the client."""

import asyncio
import time
from pathlib import Path

import pygame
import torch
from world_engine import CtrlInput, WorldEngine

from blending import blend_frames, create_blend_mask
from config import get_config, load_user_config
from seed_gen import generate_i2i, generate_t2i
from vision_api import VisionResult, describe_frame

from constants import (
    HISTORY_BROWSE_KEY,
    ImageHistoryEntry,
    i2i_executor,
    vision_executor,
)
from input import ctrl_stream
from pause_menu import show_pause_menu
from rendering import draw
from state import ClientState, GameState


async def run_loop(
    *,
    engine: WorldEngine,
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
    )

    try:
        # Apply initial game state (PAUSED - cursor visible, not grabbed)
        state.apply_game_state()

        restart = asyncio.Event()
        pause = asyncio.Event()
        limit = max(1, n_frames - 2)

        async def reset(*, reload_seed: bool = False) -> None:
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

        await reset(reload_seed=True)

        state.frames = 0
        initial_pause_done = False
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
                        # Reset engine with new T2I seed
                        state.play_time = 0.0
                        await asyncio.to_thread(engine.reset)
                        state.seed_frame = result.regenerated_frame
                        _ = await asyncio.to_thread(
                            engine.append_frame, state.seed_frame
                        )
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
                        state.last_frame = blended_frame
                        # Add to image history
                        state.image_history.insert(
                            0,
                            ImageHistoryEntry(image=blended_frame, prompt=state.prompt),
                        )

                # Start play timer when resuming
                state.play_start = time.time()
                state.game_state = GameState.PLAYING
                state.apply_game_state()
                _ = pygame.mouse.get_rel()  # discard accumulated mouse movement
                continue

            if restart.is_set() or state.frames >= limit:
                restart.clear()
                await reset(reload_seed=False)
                state.frames = 0

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

            img: torch.Tensor = await asyncio.to_thread(  # pyright: ignore[reportUnknownVariableType]
                engine.gen_frame, ctrl=filtered_ctrl
            )
            state.frames += 1
            state.last_frame = img
            draw(img, state)  # pyright: ignore[reportUnknownArgumentType]

            # Pause after first frame so engine is warm before showing pause menu
            if state.frames == 1 and not initial_pause_done:
                initial_pause_done = True
                pause.set()

            # Start i2i regeneration every i2i_interval frames (if not already running)
            if (
                i2i_interval > 0
                and state.frames > 0
                and state.frames % i2i_interval == 0
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
        pygame.event.set_grab(False)
        pygame.mixer.quit()
        pygame.quit()
