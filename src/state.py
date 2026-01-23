"""Client state management."""

from concurrent.futures import Future
from dataclasses import dataclass, field

import pygame
import torch

from constants import ImageHistoryEntry


@dataclass
class ClientState:
    """Shared state for the client, replacing closure-based state management."""

    screen: pygame.Surface
    lmb_sound: pygame.mixer.Sound
    rmb_sound: pygame.mixer.Sound

    # Core game state
    seed_frame: torch.Tensor | None = None
    prompt: str | None = None
    current_denoise: float = 0.5
    reset_time: float = 0.0
    frames: int = 0
    last_frame: torch.Tensor | None = None

    # Image history
    image_history: list[ImageHistoryEntry] = field(default_factory=list)
    history_scroll: int = 0
    history_cache: dict[int, tuple[pygame.Surface, pygame.Surface, pygame.Surface]] = (
        field(default_factory=dict)
    )

    # Async futures for background tasks
    i2i_future: Future | None = None
    i2i_pending_prompt: str | None = None
    vision_future: Future | None = None

    # Mouse button state for edge detection
    lmb_was_pressed: bool = False
    rmb_was_pressed: bool = False

    # Prompt rendering cache
    prompt_font_size: int | None = None
    cached_prompt_surface: pygame.Surface | None = None
    cached_prompt_shadow: pygame.Surface | None = None
    cached_window_width: int | None = None

    def invalidate_prompt_cache(self) -> None:
        """Clear cached prompt surfaces, forcing re-render on next draw."""
        self.cached_prompt_surface = None
        self.cached_prompt_shadow = None
        self.cached_window_width = None
