"""Client state management."""

from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import pygame
import torch

from constants import DEFAULT_BLEND_FALLOFF, ImageHistoryEntry

if TYPE_CHECKING:
    from vision_api import VisionResult


class GameState(Enum):
    """Game state for cursor/input management."""

    PAUSED = auto()  # Pause menu showing, cursor visible
    PLAYING = auto()  # Normal gameplay, cursor locked to center
    BROWSING = auto()  # Q key held, cursor visible for UI interaction


@dataclass
class ClientState:
    """Shared state for the client, replacing closure-based state management."""

    screen: pygame.Surface
    lmb_sound: pygame.mixer.Sound
    rmb_sound: pygame.mixer.Sound

    # Game state for cursor/input management
    game_state: GameState = GameState.PAUSED

    # Core game state
    seed_frame: torch.Tensor | None = None
    prompt: str | None = None
    current_denoise: float = 0.5
    play_time: float = 0.0  # Accumulated active play time
    play_start: float | None = (
        None  # When current play session started (None when paused)
    )
    frames: int = 0
    last_frame: torch.Tensor | None = None

    # Image history
    image_history: list[ImageHistoryEntry] = field(default_factory=list)
    history_scroll: int = 0
    history_cache: dict[int, tuple[pygame.Surface, pygame.Surface, pygame.Surface]] = (
        field(default_factory=dict)
    )

    # Async futures for background tasks
    i2i_future: Future[torch.Tensor] | None = None
    i2i_pending_prompt: str | None = None
    vision_future: "Future[VisionResult] | None" = None

    # VLM-triggered i2i regeneration (separate from manual RMB vision)
    vlm_i2i_vision_future: "Future[VisionResult] | None" = None

    # Mouse button state for edge detection
    lmb_was_pressed: bool = False
    rmb_was_pressed: bool = False

    # Prompt rendering cache
    prompt_font_size: int | None = None
    cached_prompt_surface: pygame.Surface | None = None
    cached_prompt_shadow: pygame.Surface | None = None
    cached_window_width: int | None = None

    # FPS tracking
    fps_last_time: float = 0.0
    fps_value: float = 0.0
    frametime_ms: float = 0.0

    # Cached blend mask (lazily created)
    blend_mask: torch.Tensor | None = None

    # User preferences (persisted to config_user.json)
    show_history_previews: bool = True
    show_prompt: bool = True
    blend_falloff: float = DEFAULT_BLEND_FALLOFF

    def invalidate_prompt_cache(self) -> None:
        """Clear cached prompt surfaces, forcing re-render on next draw."""
        self.cached_prompt_surface = None
        self.cached_prompt_shadow = None
        self.cached_window_width = None

    def apply_game_state(self) -> None:
        """Apply cursor grab/visibility based on current game state."""
        if self.game_state == GameState.PLAYING:
            pygame.event.set_grab(True)
            _ = pygame.mouse.set_visible(False)
        else:
            # PAUSED or BROWSING: cursor visible and free
            pygame.event.set_grab(False)
            _ = pygame.mouse.set_visible(True)
