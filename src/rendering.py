"""Rendering functions for the client."""

# pyright: reportUnusedCallResult=none

import time
from collections.abc import Callable, Iterable

import pygame
import torch

from constants import PROMPT_PREFIX
from engine import SLEEP_RATIO
from state import ClientState


def calculate_prompt_font_size(text: str, max_width: int, padding: int = 20) -> int:
    """Calculate font size so the prompt fits within max_width."""
    available_width = max_width - padding * 2
    for size in range(48, 8, -1):
        test_font = pygame.font.SysFont(None, size)
        text_width = test_font.size(text)[0]
        if text_width <= available_width:
            return size
    return 8


def draw_image_history(screen_surface: pygame.Surface, state: ClientState) -> None:
    """Draw scrollable image history in top-left corner."""
    if not state.image_history:
        return

    sw, sh = screen_surface.get_size()
    thumb_width = sw // 8
    thumb_height = sh // 8
    prompt_padding = 3

    # Calculate visible area
    visible_entries = 5

    # Clamp scroll offset
    max_scroll = max(0, len(state.image_history) - visible_entries)
    state.history_scroll = max(0, min(state.history_scroll, max_scroll))

    # Draw entries (most recent at top)
    y_pos = 0
    for i in range(visible_entries):
        idx = state.history_scroll + i
        if idx >= len(state.image_history):
            break
        if y_pos >= sh:
            break

        entry = state.image_history[idx]

        # Get or create cached surfaces
        entry_id = id(entry.image)
        if entry_id not in state.history_cache:
            # Create thumbnail
            img = entry.image.detach()
            if img.dtype != torch.uint8:
                img = img.clamp(0, 255).to(torch.uint8)
            frame = img.cpu().numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # pyright: ignore[reportUnknownMemberType]
            thumb = pygame.transform.scale(surf, (thumb_width, thumb_height))
            # Make thumbnail partially transparent
            thumb.set_alpha(int((80 / 100) * 255))

            # Create prompt text (sized to fit thumbnail width)
            prompt_text = entry.prompt
            if prompt_text.startswith(PROMPT_PREFIX):
                prompt_text = prompt_text[len(PROMPT_PREFIX) :]

            # Calculate font size to fit width
            font_size = 8
            for size in range(16, 7, -1):
                test_font = pygame.font.SysFont(None, size)
                if test_font.size(prompt_text)[0] <= thumb_width - prompt_padding * 2:
                    font_size = size
                    break

            prompt_font = pygame.font.SysFont(None, font_size)
            # Wrap text if needed
            words = prompt_text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if prompt_font.size(test_line)[0] <= thumb_width - prompt_padding * 2:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)  # pyright: ignore[reportUnknownMemberType]
                    current_line = word
            if current_line:
                lines.append(current_line)  # pyright: ignore[reportUnknownMemberType]

            # Render prompt lines
            line_height = prompt_font.get_height()
            prompt_surface = pygame.Surface(
                (thumb_width, line_height * len(lines) + prompt_padding * 2),  # pyright: ignore[reportUnknownArgumentType]
                pygame.SRCALPHA,
            )
            prompt_surface.fill((0, 0, 0, 150))
            for li, line in enumerate(lines):  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
                text_surf = prompt_font.render(line, True, (255, 255, 255))  # pyright: ignore[reportUnknownArgumentType]
                prompt_surface.blit(
                    text_surf,
                    (prompt_padding, prompt_padding + li * line_height),
                )

            # Shadow for prompt (unused but kept for cache structure)
            prompt_shadow = pygame.Surface(prompt_surface.get_size(), pygame.SRCALPHA)

            state.history_cache[entry_id] = (thumb, prompt_surface, prompt_shadow)

        thumb, prompt_surface, prompt_shadow = state.history_cache[entry_id]

        # Draw thumbnail (no border, no padding)
        screen_surface.blit(thumb, (0, y_pos))

        # Draw prompt below thumbnail
        prompt_y = y_pos + thumb_height
        screen_surface.blit(prompt_surface, (0, prompt_y))

        # Advance y_pos for next entry
        y_pos = prompt_y + prompt_surface.get_height()


def draw(img: torch.Tensor, state: ClientState) -> None:
    """Draw a frame to the screen with UI overlays."""
    img = img.detach()
    if img.dtype != torch.uint8:
        img = img.clamp(0, 255).to(torch.uint8)
    frame = img.cpu().numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # pyright: ignore[reportUnknownMemberType]
    surf = pygame.transform.scale(surf, state.screen.get_size())
    state.screen.blit(surf, (0, 0))

    sw, sh = state.screen.get_size()

    # Draw prompt at bottom-left (recalculate if window width changed)
    if state.prompt and state.show_prompt:
        if (
            state.cached_window_width != sw
            or state.cached_prompt_surface is None
            or state.cached_prompt_shadow is None
        ):
            state.prompt_font_size = calculate_prompt_font_size(state.prompt, sw)
            prompt_font = pygame.font.SysFont(None, state.prompt_font_size)
            state.cached_prompt_surface = prompt_font.render(
                state.prompt, True, (255, 255, 255)
            )
            state.cached_prompt_shadow = prompt_font.render(
                state.prompt, True, (0, 0, 0)
            )
            state.cached_window_width = sw
        state.screen.blit(
            state.cached_prompt_shadow,
            (11, sh - state.cached_prompt_surface.get_height() - 9),
        )
        state.screen.blit(
            state.cached_prompt_surface,
            (10, sh - state.cached_prompt_surface.get_height() - 10),
        )

    # Draw timer at top-right (stylized, only counting active play time)
    if state.play_start is not None:
        elapsed = state.play_time + (time.time() - state.play_start)
    else:
        elapsed = state.play_time
    timer_text = f"{elapsed:.1f}s"
    timer_font = pygame.font.SysFont("consolas", 36)
    timer_surface = timer_font.render(timer_text, True, (255, 255, 255))
    timer_shadow = timer_font.render(timer_text, True, (0, 0, 0))
    timer_x = sw - timer_surface.get_width() - 15
    state.screen.blit(timer_shadow, (timer_x + 2, 12))
    state.screen.blit(timer_surface, (timer_x, 10))

    # Draw denoise percentage below timer at top-right
    denoise_text = f"{int(state.current_denoise * 100)}%"
    denoise_font = pygame.font.SysFont("consolas", 28)
    denoise_surface = denoise_font.render(denoise_text, True, (200, 200, 255))
    denoise_shadow = denoise_font.render(denoise_text, True, (0, 0, 0))
    denoise_x = sw - denoise_surface.get_width() - 15
    state.screen.blit(denoise_shadow, (denoise_x + 2, 48))
    state.screen.blit(denoise_surface, (denoise_x, 46))

    # Calculate and draw FPS + frametime below denoise percentage
    current_time = time.time()
    if state.fps_last_time > 0:
        delta = current_time - state.fps_last_time
        if delta > 0:
            state.frametime_ms = delta * 1000
            state.fps_value = 1.0 / delta
    state.fps_last_time = current_time

    fps_text = f"{state.fps_value:.1f} FPS | {state.frametime_ms:.1f}ms"
    fps_font = pygame.font.SysFont("consolas", 20)
    fps_surface = fps_font.render(fps_text, True, (180, 255, 180))
    fps_shadow = fps_font.render(fps_text, True, (0, 0, 0))
    fps_x = sw - fps_surface.get_width() - 15
    state.screen.blit(fps_shadow, (fps_x + 2, 78))
    state.screen.blit(fps_surface, (fps_x, 76))

    # Draw "ANALYZING..." indicator when vision is processing
    if state.vision_future is not None:
        indicator_font = pygame.font.SysFont("consolas", 24)
        text = "ANALYZING..."
        indicator_surf = indicator_font.render(text, True, (255, 200, 100))
        shadow = indicator_font.render(text, True, (0, 0, 0))
        x = sw // 2 - indicator_surf.get_width() // 2
        state.screen.blit(shadow, (x + 2, 12))
        state.screen.blit(indicator_surf, (x, 10))

    # Draw image history in top-left
    if state.show_history_previews:
        draw_image_history(state.screen, state)

    pygame.display.flip()


def _iter_sub_frames(batch: torch.Tensor) -> list[torch.Tensor]:
    """Return a list of (H, W, 3) sub-frames from a batch tensor.

    Accepts either a single ``(H, W, 3)`` frame or a temporally-compressed
    ``(T, H, W, 3)`` batch.
    """
    if batch.dim() == 3:
        return [batch]
    return [batch[i] for i in range(batch.shape[0])]


def render_batch(
    batch: torch.Tensor,
    state: ClientState,
    pace_s: float,
    on_sub_frame: Callable[[torch.Tensor, int], None] | None = None,
) -> None:
    """Present sub-frames evenly spread over ``pace_s`` seconds.

    Uses a hybrid sleep: yields the CPU via ``pygame.time.wait`` for
    ``SLEEP_RATIO`` of the remaining interval, then busy-spins the rest for
    sub-millisecond timing precision. Sleep targets drift by up to a few
    milliseconds on most OSes, so the busy-spin closes that gap.

    The optional ``on_sub_frame`` callback runs after each draw and is given
    ``(sub_frame, sub_index)``; the game loop uses it to update
    ``state.last_frame`` / increment ``state.frames`` / record frames.
    """
    frames: Iterable[torch.Tensor] = _iter_sub_frames(batch)
    frame_list = list(frames)
    if not frame_list:
        return
    step_s = pace_s / len(frame_list)
    start = time.perf_counter()
    for i, sub in enumerate(frame_list):
        draw(sub, state)
        if on_sub_frame is not None:
            on_sub_frame(sub, i)
        deadline = start + step_s * (i + 1)
        remaining_ms = int((deadline - time.perf_counter()) * SLEEP_RATIO * 1000)
        if remaining_ms > 0:
            pygame.time.wait(remaining_ms)
        while time.perf_counter() < deadline:
            pass
