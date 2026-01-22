import argparse
import asyncio
import math
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import pygame
import torch
from world_engine import CtrlInput, WorldEngine

from config import get_config
from seed_gen import generate_i2i, generate_t2i

# Prefix to strip from prompts when displaying
PROMPT_PREFIX = "First-person view, "


def load_prompts() -> list[str]:
    """Load prompts from prompts.txt if it exists."""
    prompts_path = Path(__file__).parent.parent / "prompts.txt"
    if not prompts_path.exists():
        return []
    prompts = []
    for line in prompts_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


@dataclass
class PauseMenuResult:
    action: str  # "resume", "quit", or "regenerate"
    new_prompt: str | None = None
    regenerated_frame: torch.Tensor | None = None
    reset_with_seed: bool = False  # True = T2I reset, False = I2I append
    denoise: float = 0.5  # Denoising factor for I2I


# Separate executor for i2i so it doesn't block the engine
_i2i_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="i2i")

# pygame keycode -> Windows VK int (main ANSI rows only)
PYGAME_TO_VK = (
    {pygame.key.key_code(ch): ord(ch) for ch in "1234567890"}  # 1..0
    | {pygame.K_MINUS: 0xBD, pygame.K_EQUALS: 0xBB}  # - =
    | {pygame.key.key_code(ch): ord(ch.upper()) for ch in "qwertyuiop"}
    | {
        pygame.K_LEFTBRACKET: 0xDB,
        pygame.K_RIGHTBRACKET: 0xDD,
        pygame.K_BACKSLASH: 0xDC,
    }  # [ ] \|
    | {pygame.key.key_code(ch): ord(ch.upper()) for ch in "asdfghjkl"}
    | {pygame.K_SEMICOLON: 0xBA, pygame.K_QUOTE: 0xDE}  # ;: '"
    | {pygame.key.key_code(ch): ord(ch.upper()) for ch in "zxcvbnm"}
    | {pygame.K_COMMA: 0xBC, pygame.K_PERIOD: 0xBE, pygame.K_SLASH: 0xBF}  # ,< .> /?
    | {pygame.K_SPACE: 0x20, pygame.K_LSHIFT: 0x10, pygame.K_RSHIFT: 0x10}
)


# enable all
WHITELIST_KEYS = frozenset(PYGAME_TO_VK.values()) | frozenset({0x01, 0x02, 0x04})


async def ctrl_stream(
    restart_event: asyncio.Event,
    pause_event: asyncio.Event,
    mouse_sensitivity: float,
    whitelisted_keys=None,
) -> AsyncIterator[CtrlInput]:
    whitelisted_keys = WHITELIST_KEYS if whitelisted_keys is None else whitelisted_keys

    codes = (
        {("k", k): v for k, v in PYGAME_TO_VK.items()}
        | {
            ("m", 1): 0x01,
            ("m", 2): 0x04,
            ("m", 3): 0x02,
        }  # note: pygame has middle wheel as m2
    )
    codes = {k: v for k, v in codes.items() if v in whitelisted_keys}

    held: set[int] = set()

    while True:
        btn: set[int] = set()

        for e in pygame.event.get():  # edge presses + drain
            if e.type == pygame.QUIT:
                return

            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pause_event.set()
                elif e.key == pygame.K_u:
                    restart_event.set()

                c = codes.get(("k", e.key))
                if c is not None:
                    btn.add(c)
                    held.add(c)

            elif e.type == pygame.KEYUP:
                c = codes.get(("k", e.key))
                if c is not None:
                    held.discard(c)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                c = codes.get(("m", e.button))
                if c is not None:
                    btn.add(c)

        btn.update(held)

        mb = pygame.mouse.get_pressed(3)
        btn.update(
            c
            for i, down in enumerate(mb, 1)
            if down and (c := codes.get(("m", i))) is not None
        )

        dx, dy = pygame.mouse.get_rel()
        yield CtrlInput(
            button=btn, mouse=(dx * mouse_sensitivity, dy * mouse_sensitivity)
        )
        await asyncio.sleep(0)


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
) -> None:
    config = get_config()
    pygame.init()
    screen = pygame.display.set_mode(
        (config.window.width, config.window.height), pygame.RESIZABLE
    )
    pygame.scrap.init()  # Enable clipboard support (must be after display init)
    pygame.display.set_caption("hypnagogia (esc to pause, u to reset)")

    try:
        pygame.event.set_grab(True)

        restart = asyncio.Event()
        pause = asyncio.Event()
        ctrls = ctrl_stream(
            restart_event=restart,
            pause_event=pause,
            mouse_sensitivity=mouse_sensitivity,
        )
        limit = max(1, n_frames - 2)

        async def reset(*, reload_seed: bool = False) -> None:
            nonlocal seed_frame, reset_time
            reset_time = time.time()
            await asyncio.to_thread(engine.reset)
            if reload_seed or seed_frame is None:
                if comfyui_url and prompt:
                    seed_frame = await asyncio.to_thread(
                        generate_t2i, comfyui_url, prompt, image_seed
                    )
                else:
                    raise ValueError(
                        "ComfyUI URL and prompt are required for seed generation"
                    )
            if seed_frame is not None:
                await asyncio.to_thread(engine.append_frame, seed_frame)

        i2i_future: Future | None = None
        reset_time: float = time.time()
        current_denoise: float = config.i2i.denoise  # Track current denoise value
        prompt_font_size: int | None = None
        cached_prompt_surface: pygame.Surface | None = None
        cached_prompt_shadow: pygame.Surface | None = None
        cached_window_width: int | None = None

        def calculate_prompt_font_size(
            text: str, max_width: int, padding: int = 20
        ) -> int:
            """Calculate font size so the prompt fits within max_width."""
            available_width = max_width - padding * 2
            for size in range(48, 8, -1):
                test_font = pygame.font.SysFont(None, size)
                text_width = test_font.size(text)[0]
                if text_width <= available_width:
                    return size
            return 8

        def draw(img: torch.Tensor) -> None:
            nonlocal \
                prompt_font_size, \
                cached_prompt_surface, \
                cached_prompt_shadow, \
                cached_window_width
            img = img.detach()
            if img.dtype != torch.uint8:
                img = img.clamp(0, 255).to(torch.uint8)
            frame = img.cpu().numpy()  # (H,W,3)
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # (W,H,3)
            surf = pygame.transform.scale(surf, screen.get_size())
            screen.blit(surf, (0, 0))

            sw, sh = screen.get_size()

            # Draw prompt at bottom-left (recalculate if window width changed)
            if prompt:
                if cached_window_width != sw or cached_prompt_surface is None:
                    prompt_font_size = calculate_prompt_font_size(prompt, sw)
                    prompt_font = pygame.font.SysFont(None, prompt_font_size)
                    cached_prompt_surface = prompt_font.render(
                        prompt, True, (255, 255, 255)
                    )
                    cached_prompt_shadow = prompt_font.render(prompt, True, (0, 0, 0))
                    cached_window_width = sw
                screen.blit(
                    cached_prompt_shadow,
                    (11, sh - cached_prompt_surface.get_height() - 9),
                )
                screen.blit(
                    cached_prompt_surface,
                    (10, sh - cached_prompt_surface.get_height() - 10),
                )

            # Draw timer at top-right (stylized)
            elapsed = time.time() - reset_time
            timer_text = f"{elapsed:.1f}s"
            timer_font = pygame.font.SysFont("consolas", 36)
            timer_surface = timer_font.render(timer_text, True, (255, 255, 255))
            timer_shadow = timer_font.render(timer_text, True, (0, 0, 0))
            timer_x = sw - timer_surface.get_width() - 15
            screen.blit(timer_shadow, (timer_x + 2, 12))
            screen.blit(timer_surface, (timer_x, 10))

            pygame.display.flip()

        async def show_pause_menu() -> PauseMenuResult:
            """Show pause menu with prompt editing. Returns PauseMenuResult."""
            nonlocal prompt
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            pygame.key.set_repeat(
                400, 50
            )  # Enable key repeat (400ms delay, 50ms interval)

            # Fonts
            title_font = pygame.font.SysFont(None, 48)
            label_font = pygame.font.SysFont(None, 28)
            mono_font = pygame.font.SysFont("consolas", 18)
            button_font = pygame.font.SysFont(None, 28)
            list_font = pygame.font.SysFont("consolas", 16)

            # Capture current frame as background
            background = screen.copy()

            # Load prompts from file
            prompts_list = load_prompts()

            # Text input state
            input_text = prompt or ""
            cursor_pos = len(input_text)
            scroll_offset = 0
            input_active = False
            cursor_visible = True
            cursor_blink_time = 0.0

            # Prompts list state
            prompts_scroll = 0
            prompts_item_height = 28

            # Denoise slider state
            denoise_value = current_denoise  # Use current value, not config default
            slider_dragging = False

            # Checkbox state
            reset_checked = False

            # Menu state
            MENU, GENERATING = "menu", "generating"
            state = MENU
            gen_future: Future | None = None
            spinner_angle = 0.0
            error_message: str | None = None
            error_time = 0.0

            # UI dimensions
            input_height = 32
            input_padding = 8
            checkbox_size = 20
            button_width = 90
            button_height = 36

            # Calculate char width for monospace font
            char_width = mono_font.size("M")[0]

            while True:
                sw, sh = screen.get_size()
                dt = 1 / 60

                # Layout calculations (centered)
                center_x = sw // 2
                input_width = int(sw * 0.8)
                content_left = center_x - input_width // 2

                # Vertical layout starting from top
                title_y = 40
                prompts_list_y = title_y + 50
                prompts_list_height = min(150, max(80, sh - 380))
                visible_prompts = prompts_list_height // prompts_item_height

                input_y = prompts_list_y + prompts_list_height + 15
                slider_y = input_y + input_height + 15
                checkbox_y = slider_y + 35
                button_y = checkbox_y + 35

                input_rect = pygame.Rect(
                    content_left, input_y, input_width, input_height
                )

                # Prompts list rect
                prompts_rect = pygame.Rect(
                    content_left, prompts_list_y, input_width, prompts_list_height
                )

                # Slider dimensions
                slider_width = 200
                slider_height = 20
                slider_rect = pygame.Rect(
                    content_left, slider_y, slider_width, slider_height
                )
                slider_knob_x = slider_rect.x + int(denoise_value * slider_width)

                checkbox_rect = pygame.Rect(
                    content_left, checkbox_y, checkbox_size, checkbox_size
                )

                # Button row
                button_spacing = 12
                total_buttons_width = button_width * 4 + button_spacing * 3
                resume_rect = pygame.Rect(
                    center_x - total_buttons_width // 2,
                    button_y,
                    button_width,
                    button_height,
                )
                clear_rect = pygame.Rect(
                    resume_rect.right + button_spacing,
                    button_y,
                    button_width,
                    button_height,
                )
                submit_rect = pygame.Rect(
                    clear_rect.right + button_spacing,
                    button_y,
                    button_width,
                    button_height,
                )
                quit_rect = pygame.Rect(
                    submit_rect.right + button_spacing,
                    button_y,
                    button_width,
                    button_height,
                )

                # Event handling
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.key.set_repeat(0)
                        return PauseMenuResult(action="quit")

                    if state == MENU:
                        if e.type == pygame.KEYDOWN:
                            if e.key == pygame.K_ESCAPE:
                                pygame.key.set_repeat(0)
                                return PauseMenuResult(action="resume")

                            if input_active:
                                if e.key == pygame.K_BACKSPACE:
                                    if cursor_pos > 0:
                                        input_text = (
                                            input_text[: cursor_pos - 1]
                                            + input_text[cursor_pos:]
                                        )
                                        cursor_pos -= 1
                                elif e.key == pygame.K_DELETE:
                                    if cursor_pos < len(input_text):
                                        input_text = (
                                            input_text[:cursor_pos]
                                            + input_text[cursor_pos + 1 :]
                                        )
                                elif e.key == pygame.K_LEFT:
                                    cursor_pos = max(0, cursor_pos - 1)
                                elif e.key == pygame.K_RIGHT:
                                    cursor_pos = min(len(input_text), cursor_pos + 1)
                                elif e.key == pygame.K_HOME:
                                    cursor_pos = 0
                                elif e.key == pygame.K_END:
                                    cursor_pos = len(input_text)
                                elif e.key == pygame.K_v and (e.mod & pygame.KMOD_CTRL):
                                    try:
                                        clipboard = pygame.scrap.get(pygame.SCRAP_TEXT)
                                        if clipboard:
                                            paste_text = clipboard.decode(
                                                "utf-8"
                                            ).rstrip("\x00")
                                            input_text = (
                                                input_text[:cursor_pos]
                                                + paste_text
                                                + input_text[cursor_pos:]
                                            )
                                            cursor_pos += len(paste_text)
                                    except Exception:
                                        pass
                                elif e.key == pygame.K_RETURN:
                                    if input_text.strip():
                                        state = GENERATING
                                        if reset_checked:
                                            gen_future = _i2i_executor.submit(
                                                generate_t2i,
                                                comfyui_url,
                                                input_text,
                                                image_seed,
                                            )
                                        else:
                                            gen_future = _i2i_executor.submit(
                                                generate_i2i,
                                                comfyui_url,
                                                input_text,
                                                last_frame,
                                                None,
                                                denoise_value,
                                            )

                        if e.type == pygame.TEXTINPUT and input_active:
                            input_text = (
                                input_text[:cursor_pos]
                                + e.text
                                + input_text[cursor_pos:]
                            )
                            cursor_pos += len(e.text)

                        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                            # Check text input click
                            if input_rect.collidepoint(e.pos):
                                input_active = True
                                rel_x = e.pos[0] - input_rect.x - input_padding
                                click_char = scroll_offset + int(rel_x / char_width)
                                cursor_pos = max(0, min(len(input_text), click_char))
                            else:
                                input_active = False

                            # Check prompts list click
                            if prompts_rect.collidepoint(e.pos) and prompts_list:
                                rel_y = e.pos[1] - prompts_rect.y
                                clicked_idx = (
                                    prompts_scroll + rel_y // prompts_item_height
                                )
                                if 0 <= clicked_idx < len(prompts_list):
                                    selected_prompt = prompts_list[clicked_idx]
                                    # Strip prefix if present
                                    if selected_prompt.startswith(PROMPT_PREFIX):
                                        selected_prompt = selected_prompt[
                                            len(PROMPT_PREFIX) :
                                        ]
                                    input_text = selected_prompt
                                    cursor_pos = len(input_text)
                                    scroll_offset = 0

                            # Check slider click
                            slider_hit = pygame.Rect(
                                slider_rect.x - 5,
                                slider_rect.y - 5,
                                slider_width + 10,
                                slider_height + 10,
                            )
                            if slider_hit.collidepoint(e.pos):
                                slider_dragging = True
                                rel_x = max(
                                    0, min(slider_width, e.pos[0] - slider_rect.x)
                                )
                                denoise_value = rel_x / slider_width

                            # Check checkbox click
                            checkbox_hit = pygame.Rect(
                                checkbox_rect.x, checkbox_rect.y, 220, checkbox_size
                            )
                            if checkbox_hit.collidepoint(e.pos):
                                reset_checked = not reset_checked

                            # Check button clicks
                            if resume_rect.collidepoint(e.pos):
                                pygame.key.set_repeat(0)
                                return PauseMenuResult(action="resume")
                            if clear_rect.collidepoint(e.pos):
                                input_text = ""
                                cursor_pos = 0
                                scroll_offset = 0
                            if quit_rect.collidepoint(e.pos):
                                pygame.key.set_repeat(0)
                                return PauseMenuResult(action="quit")
                            if submit_rect.collidepoint(e.pos):
                                if input_text.strip():
                                    state = GENERATING
                                    if reset_checked:
                                        gen_future = _i2i_executor.submit(
                                            generate_t2i,
                                            comfyui_url,
                                            input_text,
                                            image_seed,
                                        )
                                    else:
                                        gen_future = _i2i_executor.submit(
                                            generate_i2i,
                                            comfyui_url,
                                            input_text,
                                            last_frame,
                                            None,
                                            denoise_value,
                                        )

                        if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                            slider_dragging = False

                        if e.type == pygame.MOUSEMOTION and slider_dragging:
                            rel_x = max(0, min(slider_width, e.pos[0] - slider_rect.x))
                            denoise_value = rel_x / slider_width

                        # Scroll prompts list with mouse wheel
                        if e.type == pygame.MOUSEWHEEL and prompts_rect.collidepoint(
                            pygame.mouse.get_pos()
                        ):
                            prompts_scroll = max(
                                0,
                                min(
                                    len(prompts_list) - visible_prompts,
                                    prompts_scroll - e.y,
                                ),
                            )

                # Check generation completion
                if state == GENERATING and gen_future is not None:
                    if gen_future.done():
                        try:
                            result_frame = gen_future.result()
                            pygame.key.set_repeat(0)
                            return PauseMenuResult(
                                action="regenerate",
                                new_prompt=input_text,
                                regenerated_frame=result_frame,
                                reset_with_seed=reset_checked,
                                denoise=denoise_value,
                            )
                        except Exception as ex:
                            error_message = f"Generation failed: {ex}"
                            error_time = time.time()
                            state = MENU
                            gen_future = None
                            print(f"Generation error: {ex}")

                # Clear error after 3 seconds
                if error_message and time.time() - error_time > 3.0:
                    error_message = None

                # Update cursor blink
                cursor_blink_time += dt
                if cursor_blink_time >= 0.5:
                    cursor_blink_time = 0.0
                    cursor_visible = not cursor_visible

                # Update spinner
                spinner_angle += dt * 360  # One rotation per second

                # Calculate scroll offset to keep cursor visible
                visible_chars = (input_width - 2 * input_padding) // char_width
                if cursor_pos < scroll_offset:
                    scroll_offset = cursor_pos
                elif cursor_pos >= scroll_offset + visible_chars:
                    scroll_offset = cursor_pos - visible_chars + 1
                scroll_offset = max(0, scroll_offset)

                # --- Drawing ---
                screen.blit(background, (0, 0))

                # Semi-transparent overlay
                overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                screen.blit(overlay, (0, 0))

                # Title
                title = title_font.render("PAUSED", True, (255, 255, 255))
                screen.blit(title, (center_x - title.get_width() // 2, title_y))

                # --- Prompts List ---
                if prompts_list:
                    prompts_label = label_font.render("Prompts:", True, (255, 255, 255))
                    screen.blit(prompts_label, (content_left, prompts_list_y - 22))

                    # List background
                    pygame.draw.rect(
                        screen, (40, 40, 40), prompts_rect, border_radius=4
                    )
                    pygame.draw.rect(
                        screen, (80, 80, 80), prompts_rect, 1, border_radius=4
                    )

                    # Draw visible prompts with clipping
                    screen.set_clip(prompts_rect)
                    mouse_pos = pygame.mouse.get_pos()
                    for i in range(visible_prompts + 1):
                        idx = prompts_scroll + i
                        if idx >= len(prompts_list):
                            break
                        item_y = prompts_rect.y + i * prompts_item_height
                        item_rect = pygame.Rect(
                            prompts_rect.x,
                            item_y,
                            prompts_rect.width,
                            prompts_item_height,
                        )

                        # Highlight on hover
                        if item_rect.collidepoint(mouse_pos):
                            pygame.draw.rect(screen, (60, 60, 80), item_rect)

                        # Display prompt (strip prefix)
                        display_text = prompts_list[idx]
                        if display_text.startswith(PROMPT_PREFIX):
                            display_text = display_text[len(PROMPT_PREFIX) :]

                        # Truncate if too long
                        max_chars = (input_width - 20) // list_font.size("M")[0]
                        if len(display_text) > max_chars:
                            display_text = display_text[: max_chars - 3] + "..."

                        text_surf = list_font.render(
                            display_text, True, (200, 200, 200)
                        )
                        screen.blit(
                            text_surf,
                            (
                                prompts_rect.x + 8,
                                item_y
                                + (prompts_item_height - text_surf.get_height()) // 2,
                            ),
                        )
                    screen.set_clip(None)

                    # Scrollbar (if needed)
                    if len(prompts_list) > visible_prompts:
                        scrollbar_height = max(
                            20,
                            prompts_list_height * visible_prompts // len(prompts_list),
                        )
                        scrollbar_y = prompts_rect.y + int(
                            (prompts_list_height - scrollbar_height)
                            * prompts_scroll
                            / max(1, len(prompts_list) - visible_prompts)
                        )
                        scrollbar_rect = pygame.Rect(
                            prompts_rect.right - 8, scrollbar_y, 6, scrollbar_height
                        )
                        pygame.draw.rect(
                            screen, (100, 100, 100), scrollbar_rect, border_radius=3
                        )

                # --- Prompt Input ---
                prompt_label = label_font.render("Prompt:", True, (255, 255, 255))
                screen.blit(prompt_label, (input_rect.x, input_rect.y - 22))

                input_color = (80, 80, 80) if input_active else (50, 50, 50)
                border_color = (150, 150, 255) if input_active else (100, 100, 100)
                pygame.draw.rect(screen, input_color, input_rect, border_radius=4)
                pygame.draw.rect(screen, border_color, input_rect, 2, border_radius=4)

                # Render visible text with clipping
                clip_rect = pygame.Rect(
                    input_rect.x + input_padding,
                    input_rect.y,
                    input_width - 2 * input_padding,
                    input_height,
                )
                visible_text = input_text[
                    scroll_offset : scroll_offset + visible_chars + 1
                ]
                text_surface = mono_font.render(visible_text, True, (255, 255, 255))
                text_y = input_rect.y + (input_height - text_surface.get_height()) // 2

                screen.set_clip(clip_rect)
                screen.blit(text_surface, (input_rect.x + input_padding, text_y))
                screen.set_clip(None)

                # Draw cursor
                if input_active and cursor_visible:
                    cursor_x = (
                        input_rect.x
                        + input_padding
                        + (cursor_pos - scroll_offset) * char_width
                    )
                    pygame.draw.line(
                        screen,
                        (255, 255, 255),
                        (cursor_x, input_rect.y + 4),
                        (cursor_x, input_rect.y + input_height - 4),
                        2,
                    )

                # --- Denoise Slider ---
                slider_label = label_font.render(
                    f"Denoise: {denoise_value:.2f}", True, (255, 255, 255)
                )
                screen.blit(slider_label, (slider_rect.right + 15, slider_rect.y))

                # Slider track
                pygame.draw.rect(screen, (60, 60, 60), slider_rect, border_radius=4)
                # Filled portion
                filled_rect = pygame.Rect(
                    slider_rect.x,
                    slider_rect.y,
                    int(denoise_value * slider_width),
                    slider_height,
                )
                pygame.draw.rect(screen, (100, 100, 180), filled_rect, border_radius=4)
                # Slider knob
                knob_rect = pygame.Rect(
                    slider_rect.x + int(denoise_value * slider_width) - 6,
                    slider_rect.y - 2,
                    12,
                    slider_height + 4,
                )
                pygame.draw.rect(screen, (200, 200, 200), knob_rect, border_radius=4)

                # --- Checkbox ---
                pygame.draw.rect(screen, (50, 50, 50), checkbox_rect, border_radius=4)
                pygame.draw.rect(
                    screen, (100, 100, 100), checkbox_rect, 2, border_radius=4
                )
                if reset_checked:
                    inner = checkbox_rect.inflate(-6, -6)
                    pygame.draw.rect(screen, (100, 200, 100), inner, border_radius=2)

                checkbox_label = label_font.render(
                    "Reset (new seed)", True, (255, 255, 255)
                )
                screen.blit(
                    checkbox_label,
                    (
                        checkbox_rect.right + 10,
                        checkbox_rect.centery - checkbox_label.get_height() // 2,
                    ),
                )

                # Buttons
                mouse_pos = pygame.mouse.get_pos()
                buttons = [
                    (resume_rect, "Resume"),
                    (clear_rect, "Clear"),
                    (submit_rect, "Submit"),
                    (quit_rect, "Quit"),
                ]
                for rect, text in buttons:
                    is_submit = text == "Submit"
                    is_hovered = rect.collidepoint(mouse_pos)

                    if state == GENERATING and is_submit:
                        color = (40, 40, 40)  # Disabled
                    elif is_hovered:
                        color = (100, 100, 100)
                    else:
                        color = (60, 60, 60)

                    pygame.draw.rect(screen, color, rect, border_radius=8)
                    pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=8)

                    label = button_font.render(text, True, (255, 255, 255))
                    screen.blit(
                        label,
                        (
                            rect.centerx - label.get_width() // 2,
                            rect.centery - label.get_height() // 2,
                        ),
                    )

                # Spinner during generation
                if state == GENERATING:
                    spinner_x = center_x
                    spinner_y = button_y + button_height + 40
                    spinner_radius = 20

                    # Draw rotating arc
                    start_angle = math.radians(spinner_angle)
                    end_angle = start_angle + math.radians(270)
                    arc_rect = pygame.Rect(
                        spinner_x - spinner_radius,
                        spinner_y - spinner_radius,
                        spinner_radius * 2,
                        spinner_radius * 2,
                    )
                    pygame.draw.arc(
                        screen,
                        (150, 150, 255),
                        arc_rect,
                        start_angle,
                        end_angle,
                        4,
                    )

                    gen_label = label_font.render(
                        "Generating...", True, (200, 200, 200)
                    )
                    screen.blit(
                        gen_label,
                        (center_x - gen_label.get_width() // 2, spinner_y + 30),
                    )

                # Error message
                if error_message:
                    error_surface = label_font.render(
                        error_message, True, (255, 100, 100)
                    )
                    screen.blit(
                        error_surface,
                        (
                            center_x - error_surface.get_width() // 2,
                            button_y + button_height + 20,
                        ),
                    )

                pygame.display.flip()
                await asyncio.sleep(1 / 60)

        await reset(reload_seed=True)

        frames = 0
        last_frame: torch.Tensor | None = None
        async for ctrl in ctrls:
            # Check if i2i task completed (non-blocking)
            if i2i_future is not None and i2i_future.done():
                refreshed = i2i_future.result()
                await asyncio.to_thread(engine.append_frame, refreshed)
                i2i_future = None

            if pause.is_set():
                pause.clear()
                result = await show_pause_menu()

                if result.action == "quit":
                    return

                if result.action == "regenerate":
                    prompt = result.new_prompt  # Update prompt
                    current_denoise = result.denoise  # Update denoise value
                    # Invalidate cached prompt surfaces
                    cached_prompt_surface = None
                    cached_prompt_shadow = None
                    cached_window_width = None

                    if result.reset_with_seed:
                        # Reset engine with new T2I seed
                        reset_time = time.time()
                        await asyncio.to_thread(engine.reset)
                        seed_frame = result.regenerated_frame
                        await asyncio.to_thread(engine.append_frame, seed_frame)
                        frames = 0
                    else:
                        # Append I2I regenerated frame and continue
                        await asyncio.to_thread(
                            engine.append_frame, result.regenerated_frame
                        )
                        last_frame = result.regenerated_frame

                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                pygame.mouse.get_rel()  # discard accumulated mouse movement
                continue

            if restart.is_set() or frames >= limit:
                restart.clear()
                await reset(reload_seed=False)
                frames = 0

            img = await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
            frames += 1
            last_frame = img
            draw(img)

            # Start i2i regeneration every i2i_interval frames (if not already running)
            if (
                i2i_interval > 0
                and frames > 0
                and frames % i2i_interval == 0
                and i2i_future is None
                and comfyui_url
                and prompt
            ):
                i2i_future = _i2i_executor.submit(
                    generate_i2i, comfyui_url, prompt, last_frame, None, current_denoise
                )

            await asyncio.sleep(0)
    finally:
        pygame.event.set_grab(False)
        pygame.quit()


async def main(
    *,
    comfyui_url: str,
    prompt: str,
    image_seed: int | None = None,
    n_frames: int,
    device: str,
    i2i_interval: int,
    mouse_sensitivity: float,
) -> None:
    config = get_config()
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

    def _cuda_warmup() -> None:
        with torch.cuda.device(device):
            torch.cuda.current_blas_handle()

    await asyncio.to_thread(_cuda_warmup)

    engine = WorldEngine(
        config.models.world_engine,
        device=device,
        model_config_overrides={
            "n_frames": n_frames,
            "ae_uri": config.models.vae_uri,
        },
    )
    await run_loop(
        engine=engine,
        seed_frame=None,
        n_frames=n_frames,
        mouse_sensitivity=mouse_sensitivity,
        comfyui_url=comfyui_url,
        prompt=prompt,
        image_seed=image_seed,
        i2i_interval=i2i_interval,
    )


def cli() -> None:
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Local World client with ComfyUI seed generation"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="ComfyUI server URL (e.g., http://127.0.0.1:8188)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt for seed image generation (default: random from prompts.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for image generation (default: random)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=config.defaults.n_frames,
        help=f"Number of frames (default: {config.defaults.n_frames})",
    )
    parser.add_argument(
        "--i2i-interval",
        type=int,
        default=config.defaults.i2i_interval,
        help=f"Frames between i2i regeneration (default: {config.defaults.i2i_interval}, 0 to disable)",
    )
    parser.add_argument(
        "--device",
        default=config.defaults.device,
        help=f"Device to use (default: {config.defaults.device})",
    )
    parser.add_argument(
        "--mouse-sensitivity",
        type=float,
        default=config.defaults.mouse_sensitivity,
        help=f"Mouse sensitivity (default: {config.defaults.mouse_sensitivity})",
    )
    args = parser.parse_args()

    # Pick random prompt from prompts.txt if not specified
    prompt = args.prompt
    if prompt is None:
        prompts = load_prompts()
        if prompts:
            prompt = random.choice(prompts)
            print(f"Using random prompt: {prompt}")
        else:
            parser.error("--prompt is required (no prompts.txt found)")

    asyncio.run(
        main(
            comfyui_url=args.url,
            prompt=prompt,
            image_seed=args.seed,
            n_frames=args.n_frames,
            device=args.device,
            i2i_interval=args.i2i_interval,
            mouse_sensitivity=args.mouse_sensitivity,
        )
    )


if __name__ == "__main__":
    cli()
