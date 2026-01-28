"""Pause menu implementation."""

# pyright: reportUnusedCallResult=none

import asyncio
import math
import time
from concurrent.futures import Future

import pygame
import torch

from blending import create_blend_mask
from seed_gen import ENGINE_RESOLUTION, generate_i2i, generate_t2i

from config import UserConfig, save_user_config
from constants import PROMPT_PREFIX, PauseMenuResult, i2i_executor, load_prompts
from state import ClientState


async def show_pause_menu(
    screen: pygame.Surface,
    state: ClientState,
    last_frame: torch.Tensor | None,
    comfyui_url: str | None,
    image_seed: int | None,
) -> PauseMenuResult:
    """Show pause menu with prompt editing. Returns PauseMenuResult."""
    pygame.event.set_grab(False)
    _ = pygame.mouse.set_visible(True)
    _ = pygame.key.set_repeat(400, 50)  # Enable key repeat (400ms delay, 50ms interval)

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
    input_text = state.prompt or ""
    cursor_pos = len(input_text)
    scroll_offset = 0
    input_active = False
    cursor_visible = True
    cursor_blink_time = 0.0

    # Prompts list state
    prompts_scroll = 0
    prompts_item_height = 28

    # Denoise slider state
    denoise_value = state.current_denoise  # Use current value, not config default
    slider_dragging = False

    # Blend falloff slider state
    blend_falloff_value = state.blend_falloff
    blend_falloff_dragging = False
    mask_preview_surface: pygame.Surface | None = None

    # Checkbox state
    reset_checked = False
    show_history_checked = state.show_history_previews
    show_prompt_checked = state.show_prompt

    # Menu state
    MENU, GENERATING = "menu", "generating"
    menu_state = MENU
    gen_future: Future[torch.Tensor] | None = None
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
        prompts_list_height = min(150, max(80, sh - 420))  # Adjusted for blend slider
        visible_prompts = prompts_list_height // prompts_item_height

        input_y = prompts_list_y + prompts_list_height + 15
        slider_y = input_y + input_height + 15
        blend_slider_y = slider_y + 35
        checkbox_y = blend_slider_y + 35
        checkbox2_y = checkbox_y + 28
        checkbox3_y = checkbox2_y + 28
        button_y = checkbox3_y + 35

        input_rect = pygame.Rect(content_left, input_y, input_width, input_height)

        # Prompts list rect
        prompts_rect = pygame.Rect(
            content_left, prompts_list_y, input_width, prompts_list_height
        )

        # Slider dimensions
        slider_width = 200
        slider_height = 20
        slider_rect = pygame.Rect(content_left, slider_y, slider_width, slider_height)
        blend_slider_rect = pygame.Rect(
            content_left, blend_slider_y, slider_width, slider_height
        )

        checkbox_rect = pygame.Rect(
            content_left, checkbox_y, checkbox_size, checkbox_size
        )
        checkbox2_rect = pygame.Rect(
            content_left, checkbox2_y, checkbox_size, checkbox_size
        )
        checkbox3_rect = pygame.Rect(
            content_left, checkbox3_y, checkbox_size, checkbox_size
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

            if menu_state == MENU:
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:  # pyright: ignore[reportAny]
                        pygame.key.set_repeat(0)
                        return PauseMenuResult(action="resume")

                    if input_active:
                        if e.key == pygame.K_BACKSPACE:  # pyright: ignore[reportAny]
                            if cursor_pos > 0:
                                input_text = (
                                    input_text[: cursor_pos - 1]
                                    + input_text[cursor_pos:]
                                )
                                cursor_pos -= 1
                        elif e.key == pygame.K_DELETE:  # pyright: ignore[reportAny]
                            if cursor_pos < len(input_text):
                                input_text = (
                                    input_text[:cursor_pos]
                                    + input_text[cursor_pos + 1 :]
                                )
                        elif e.key == pygame.K_LEFT:  # pyright: ignore[reportAny]
                            cursor_pos = max(0, cursor_pos - 1)
                        elif e.key == pygame.K_RIGHT:  # pyright: ignore[reportAny]
                            cursor_pos = min(len(input_text), cursor_pos + 1)
                        elif e.key == pygame.K_HOME:  # pyright: ignore[reportAny]
                            cursor_pos = 0
                        elif e.key == pygame.K_END:  # pyright: ignore[reportAny]
                            cursor_pos = len(input_text)
                        elif e.key == pygame.K_v and (e.mod & pygame.KMOD_CTRL):  # pyright: ignore[reportAny]
                            try:
                                clipboard = pygame.scrap.get(pygame.SCRAP_TEXT)
                                if clipboard:
                                    paste_text = clipboard.decode("utf-8").rstrip(
                                        "\x00"
                                    )
                                    input_text = (
                                        input_text[:cursor_pos]
                                        + paste_text
                                        + input_text[cursor_pos:]
                                    )
                                    cursor_pos += len(paste_text)
                            except Exception:
                                pass
                        elif e.key == pygame.K_RETURN:  # pyright: ignore[reportAny]
                            if input_text.strip() and comfyui_url is not None:
                                menu_state = GENERATING
                                if reset_checked:
                                    gen_future = i2i_executor.submit(
                                        generate_t2i,
                                        comfyui_url,
                                        input_text,
                                        image_seed,
                                    )
                                elif last_frame is not None:
                                    gen_future = i2i_executor.submit(
                                        generate_i2i,
                                        comfyui_url,
                                        input_text,
                                        last_frame,
                                        None,
                                        denoise_value,
                                    )

                if e.type == pygame.TEXTINPUT and input_active:
                    input_text = (  # pyright: ignore[reportAny]
                        input_text[:cursor_pos] + e.text + input_text[cursor_pos:]  # pyright: ignore[reportAny]
                    )
                    cursor_pos += len(e.text)  # pyright: ignore[reportAny]

                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:  # pyright: ignore[reportAny]
                    # Check text input click
                    if input_rect.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        input_active = True
                        rel_x: int = e.pos[0] - input_rect.x - input_padding  # pyright: ignore[reportAny]
                        click_char = scroll_offset + int(rel_x / char_width)
                        cursor_pos = max(0, min(len(input_text), click_char))
                    else:
                        input_active = False

                    # Check prompts list click
                    if prompts_rect.collidepoint(e.pos) and prompts_list:  # pyright: ignore[reportAny]
                        rel_y: int = e.pos[1] - prompts_rect.y  # pyright: ignore[reportAny]
                        clicked_idx: int = prompts_scroll + rel_y // prompts_item_height
                        if 0 <= clicked_idx < len(prompts_list):
                            selected_prompt: str = prompts_list[clicked_idx]
                            # Strip prefix if present
                            if selected_prompt.startswith(PROMPT_PREFIX):
                                selected_prompt = selected_prompt[len(PROMPT_PREFIX) :]
                            input_text = selected_prompt
                            cursor_pos = len(input_text)
                            scroll_offset = 0

                    # Check denoise slider click
                    slider_hit = pygame.Rect(
                        slider_rect.x - 5,
                        slider_rect.y - 5,
                        slider_width + 10,
                        slider_height + 10,
                    )
                    if slider_hit.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        slider_dragging = True
                        rel_x = max(0, min(slider_width, e.pos[0] - slider_rect.x))  # pyright: ignore[reportAny]
                        denoise_value = rel_x / slider_width

                    # Check blend falloff slider click
                    blend_slider_hit = pygame.Rect(
                        blend_slider_rect.x - 5,
                        blend_slider_rect.y - 5,
                        slider_width + 10,
                        slider_height + 10,
                    )
                    if blend_slider_hit.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        blend_falloff_dragging = True
                        rel_x = max(
                            0,
                            min(slider_width, e.pos[0] - blend_slider_rect.x),  # pyright: ignore[reportAny]
                        )
                        blend_falloff_value = rel_x / slider_width
                        mask_preview_surface = None  # Invalidate preview
                        state.blend_falloff = blend_falloff_value
                        state.blend_mask = None  # Invalidate mask cache
                        save_user_config(
                            UserConfig(
                                show_history_previews=show_history_checked,
                                show_prompt=show_prompt_checked,
                                blend_falloff=blend_falloff_value,
                            )
                        )

                    # Check checkbox clicks
                    checkbox_hit = pygame.Rect(
                        checkbox_rect.x, checkbox_rect.y, 220, checkbox_size
                    )
                    if checkbox_hit.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        reset_checked = not reset_checked

                    checkbox2_hit = pygame.Rect(
                        checkbox2_rect.x, checkbox2_rect.y, 220, checkbox_size
                    )
                    if checkbox2_hit.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        show_history_checked = not show_history_checked
                        state.show_history_previews = show_history_checked
                        save_user_config(
                            UserConfig(
                                show_history_previews=show_history_checked,
                                show_prompt=show_prompt_checked,
                                blend_falloff=blend_falloff_value,
                            )
                        )

                    checkbox3_hit = pygame.Rect(
                        checkbox3_rect.x, checkbox3_rect.y, 220, checkbox_size
                    )
                    if checkbox3_hit.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        show_prompt_checked = not show_prompt_checked
                        state.show_prompt = show_prompt_checked
                        save_user_config(
                            UserConfig(
                                show_history_previews=show_history_checked,
                                show_prompt=show_prompt_checked,
                                blend_falloff=blend_falloff_value,
                            )
                        )

                    # Check button clicks
                    if resume_rect.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        pygame.key.set_repeat(0)
                        return PauseMenuResult(action="resume")
                    if clear_rect.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        input_text = ""
                        cursor_pos = 0
                        scroll_offset = 0
                    if quit_rect.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        pygame.key.set_repeat(0)
                        return PauseMenuResult(action="quit")
                    if submit_rect.collidepoint(e.pos):  # pyright: ignore[reportAny]
                        if input_text.strip() and comfyui_url is not None:
                            menu_state = GENERATING
                            if reset_checked:
                                gen_future = i2i_executor.submit(
                                    generate_t2i,
                                    comfyui_url,
                                    input_text,
                                    image_seed,
                                )
                            elif last_frame is not None:
                                gen_future = i2i_executor.submit(
                                    generate_i2i,
                                    comfyui_url,
                                    input_text,
                                    last_frame,
                                    None,
                                    denoise_value,
                                )

                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    slider_dragging = False
                    blend_falloff_dragging = False

                if e.type == pygame.MOUSEMOTION:
                    if slider_dragging:
                        rel_x = max(0, min(slider_width, e.pos[0] - slider_rect.x))  # pyright: ignore[reportAny]
                        denoise_value = rel_x / slider_width
                    if blend_falloff_dragging:
                        rel_x = max(
                            0,
                            min(slider_width, e.pos[0] - blend_slider_rect.x),  # pyright: ignore[reportAny]
                        )
                        blend_falloff_value = rel_x / slider_width
                        mask_preview_surface = None  # Invalidate preview
                        state.blend_falloff = blend_falloff_value
                        state.blend_mask = None  # Invalidate mask cache
                        save_user_config(
                            UserConfig(
                                show_history_previews=show_history_checked,
                                show_prompt=show_prompt_checked,
                                blend_falloff=blend_falloff_value,
                            )
                        )

                # Scroll prompts list with mouse wheel
                if e.type == pygame.MOUSEWHEEL and prompts_rect.collidepoint(
                    pygame.mouse.get_pos()
                ):
                    prompts_scroll = max(
                        0,
                        min(
                            len(prompts_list) - visible_prompts,
                            prompts_scroll - e.y,  # pyright: ignore[reportAny]
                        ),
                    )

        # Check generation completion
        if menu_state == GENERATING and gen_future is not None:
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
                    menu_state = MENU
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
            pygame.draw.rect(screen, (40, 40, 40), prompts_rect, border_radius=4)
            pygame.draw.rect(screen, (80, 80, 80), prompts_rect, 1, border_radius=4)

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

                text_surf = list_font.render(display_text, True, (200, 200, 200))
                screen.blit(
                    text_surf,
                    (
                        prompts_rect.x + 8,
                        item_y + (prompts_item_height - text_surf.get_height()) // 2,
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
        visible_text = input_text[scroll_offset : scroll_offset + visible_chars + 1]
        text_surface = mono_font.render(visible_text, True, (255, 255, 255))
        text_y = input_rect.y + (input_height - text_surface.get_height()) // 2

        screen.set_clip(clip_rect)
        screen.blit(text_surface, (input_rect.x + input_padding, text_y))
        screen.set_clip(None)

        # Draw cursor
        if input_active and cursor_visible:
            cursor_x = (
                input_rect.x + input_padding + (cursor_pos - scroll_offset) * char_width
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

        # --- Blend Falloff Slider ---
        blend_slider_label = label_font.render(
            f"Blend Falloff: {blend_falloff_value:.2f}", True, (255, 255, 255)
        )
        label_x = blend_slider_rect.right + 15
        label_y = blend_slider_rect.y
        screen.blit(blend_slider_label, (label_x, label_y))

        # Slider track
        pygame.draw.rect(screen, (60, 60, 60), blend_slider_rect, border_radius=4)
        # Filled portion
        blend_filled_rect = pygame.Rect(
            blend_slider_rect.x,
            blend_slider_rect.y,
            int(blend_falloff_value * slider_width),
            slider_height,
        )
        pygame.draw.rect(screen, (100, 180, 100), blend_filled_rect, border_radius=4)
        # Slider knob
        blend_knob_rect = pygame.Rect(
            blend_slider_rect.x + int(blend_falloff_value * slider_width) - 6,
            blend_slider_rect.y - 2,
            12,
            slider_height + 4,
        )
        pygame.draw.rect(screen, (200, 200, 200), blend_knob_rect, border_radius=4)

        # --- Mask Preview ---
        # Calculate preview dimensions: same height as label, ENGINE_RESOLUTION aspect ratio
        preview_height = blend_slider_label.get_height()
        aspect_ratio = ENGINE_RESOLUTION[0] / ENGINE_RESOLUTION[1]  # width / height
        preview_width = int(preview_height * aspect_ratio)
        preview_x = label_x + blend_slider_label.get_width() + 10
        preview_y = label_y

        # Generate mask preview if needed (or if size changed)
        if (
            mask_preview_surface is None
            or mask_preview_surface.get_width() != preview_width
            or mask_preview_surface.get_height() != preview_height
        ):
            mask_tensor = create_blend_mask(
                preview_height, preview_width, blend_falloff_value
            )
            # Convert to 8-bit grayscale and then to pygame surface
            mask_uint8 = (mask_tensor * 255).to(torch.uint8).numpy()  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
            # Create RGB surface from grayscale
            mask_rgb = mask_uint8[:, :, None].repeat(3, axis=2)  # pyright: ignore[reportUnknownVariableType]
            mask_preview_surface = pygame.surfarray.make_surface(  # pyright: ignore[reportUnknownMemberType]
                mask_rgb.transpose(1, 0, 2)
            )

        # Draw mask preview with border
        preview_rect = pygame.Rect(preview_x, preview_y, preview_width, preview_height)
        pygame.draw.rect(
            screen, (60, 60, 60), preview_rect.inflate(4, 4), border_radius=2
        )
        screen.blit(mask_preview_surface, preview_rect.topleft)
        pygame.draw.rect(screen, (100, 100, 100), preview_rect, 1, border_radius=1)

        # --- Checkboxes ---
        # Reset checkbox
        pygame.draw.rect(screen, (50, 50, 50), checkbox_rect, border_radius=4)
        pygame.draw.rect(screen, (100, 100, 100), checkbox_rect, 2, border_radius=4)
        if reset_checked:
            inner = checkbox_rect.inflate(-6, -6)
            pygame.draw.rect(screen, (100, 200, 100), inner, border_radius=2)

        checkbox_label = label_font.render("Reset (new seed)", True, (255, 255, 255))
        screen.blit(
            checkbox_label,
            (
                checkbox_rect.right + 10,
                checkbox_rect.centery - checkbox_label.get_height() // 2,
            ),
        )

        # Show history previews checkbox
        pygame.draw.rect(screen, (50, 50, 50), checkbox2_rect, border_radius=4)
        pygame.draw.rect(screen, (100, 100, 100), checkbox2_rect, 2, border_radius=4)
        if show_history_checked:
            inner2 = checkbox2_rect.inflate(-6, -6)
            pygame.draw.rect(screen, (100, 200, 100), inner2, border_radius=2)

        checkbox2_label = label_font.render(
            "Show history previews", True, (255, 255, 255)
        )
        screen.blit(
            checkbox2_label,
            (
                checkbox2_rect.right + 10,
                checkbox2_rect.centery - checkbox2_label.get_height() // 2,
            ),
        )

        # Show prompt checkbox
        pygame.draw.rect(screen, (50, 50, 50), checkbox3_rect, border_radius=4)
        pygame.draw.rect(screen, (100, 100, 100), checkbox3_rect, 2, border_radius=4)
        if show_prompt_checked:
            inner3 = checkbox3_rect.inflate(-6, -6)
            pygame.draw.rect(screen, (100, 200, 100), inner3, border_radius=2)

        checkbox3_label = label_font.render("Show prompt", True, (255, 255, 255))
        screen.blit(
            checkbox3_label,
            (
                checkbox3_rect.right + 10,
                checkbox3_rect.centery - checkbox3_label.get_height() // 2,
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

            if menu_state == GENERATING and is_submit:
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
        if menu_state == GENERATING:
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

            gen_label = label_font.render("Generating...", True, (200, 200, 200))
            screen.blit(
                gen_label,
                (center_x - gen_label.get_width() // 2, spinner_y + 30),
            )

        # Error message
        if error_message:
            error_surface = label_font.render(error_message, True, (255, 100, 100))
            screen.blit(
                error_surface,
                (
                    center_x - error_surface.get_width() // 2,
                    button_y + button_height + 20,
                ),
            )

        pygame.display.flip()
        await asyncio.sleep(1 / 60)
