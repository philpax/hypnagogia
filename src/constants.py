"""Constants, data classes, and shared utilities for the client."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pygame
import torch


# Prefix to strip from prompts when displaying
PROMPT_PREFIX = "First-person view, "

# Key to hold for browsing image history (releases mouse grab)
HISTORY_BROWSE_KEY = pygame.K_q

# Default blend mask falloff threshold (0.0 to 1.0)
DEFAULT_BLEND_FALLOFF = 0.5


def load_prompts() -> list[str]:
    """Load prompts from prompts.txt if it exists."""
    prompts_path = Path(__file__).parent.parent / "prompts.txt"
    if not prompts_path.exists():
        return []
    prompts: list[str] = []
    for line in prompts_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


@dataclass
class PauseMenuResult:
    action: str  # "resume", "quit", "regenerate", "replay", "rerecord", or "rerecord_primed"
    new_prompt: str | None = None
    regenerated_frame: torch.Tensor | None = None
    reset_with_seed: bool = False  # True = T2I reset, False = I2I append
    denoise: float = 0.5  # Denoising factor for I2I
    replay_json_path: Path | None = None


@dataclass
class ImageHistoryEntry:
    """An entry in the image history showing T2I/I2I generations."""

    image: torch.Tensor  # The generated image
    prompt: str  # The prompt used to generate it


# Separate executor for i2i so it doesn't block the engine
i2i_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="i2i")

# Separate executor for vision API calls
vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")

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
