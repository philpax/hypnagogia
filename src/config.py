"""Configuration loader for local_world."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

from constants import DEFAULT_BLEND_FALLOFF


class WindowConfigDict(TypedDict):
    width: int
    height: int


class DefaultsConfigDict(TypedDict):
    n_frames: int
    i2i_interval: int
    i2i_vlm_regen: bool
    device: str
    mouse_sensitivity: float
    quant: str | None


class ModelsConfigDict(TypedDict, total=False):
    world_engine: str
    vae_uri: str


class T2IConfigDict(TypedDict):
    clip: str
    vae: str
    unet: str
    sampler: str
    steps: int
    cfg: int


class I2IConfigDict(TypedDict):
    checkpoint: str
    sampler: str
    steps: int
    denoise: float


class VisionConfigDict(TypedDict):
    api_url: str
    model: str
    api_key_env: str
    max_tokens: int
    timeout: float


class ConfigDict(TypedDict):
    window: WindowConfigDict
    defaults: DefaultsConfigDict
    models: ModelsConfigDict
    t2i: T2IConfigDict
    i2i: I2IConfigDict
    vision: VisionConfigDict


@dataclass
class WindowConfig:
    width: int
    height: int


@dataclass
class DefaultsConfig:
    n_frames: int
    i2i_interval: int
    i2i_vlm_regen: bool
    device: str
    mouse_sensitivity: float
    quant: str | None = None


@dataclass
class ModelsConfig:
    world_engine: str
    vae_uri: str | None = None


@dataclass
class T2IConfig:
    clip: str
    vae: str
    unet: str
    sampler: str
    steps: int
    cfg: int


@dataclass
class I2IConfig:
    checkpoint: str
    sampler: str
    steps: int
    denoise: float


@dataclass
class VisionConfig:
    api_url: str
    model: str
    api_key_env: str
    max_tokens: int
    timeout: float


@dataclass
class Config:
    window: WindowConfig
    defaults: DefaultsConfig
    models: ModelsConfig
    t2i: T2IConfig
    i2i: I2IConfig
    vision: VisionConfig


def load_config() -> Config:
    """Load configuration from config.json in project root."""
    # Find config.json relative to this file (src/config.py -> ../config.json)
    config_path = Path(__file__).parent.parent / "config.json"

    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_path) as f:
            data = cast(ConfigDict, json.load(f))
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config.json: {e}", file=sys.stderr)
        sys.exit(1)

    return Config(
        window=WindowConfig(**data["window"]),
        defaults=DefaultsConfig(**data["defaults"]),
        models=ModelsConfig(**data["models"]),
        t2i=T2IConfig(**data["t2i"]),
        i2i=I2IConfig(**data["i2i"]),
        vision=VisionConfig(**data["vision"]),
    )


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


class UserConfigDict(TypedDict, total=False):
    show_history_previews: bool
    show_prompt: bool
    blend_falloff: float
    click_repainting: bool
    recording_enabled: bool


@dataclass
class UserConfig:
    """User-specific configuration that persists across sessions."""

    show_history_previews: bool = field(default=True)
    show_prompt: bool = field(default=True)
    blend_falloff: float = field(default=DEFAULT_BLEND_FALLOFF)
    click_repainting: bool = field(default=True)
    recording_enabled: bool = field(default=False)


def _get_user_config_path() -> Path:
    """Get the path to config_user.json."""
    return Path(__file__).parent.parent / "config_user.json"


def load_user_config() -> UserConfig:
    """Load user configuration from config_user.json, creating defaults if missing."""
    config_path = _get_user_config_path()

    if not config_path.exists():
        return UserConfig()

    try:
        with open(config_path) as f:
            data = cast(UserConfigDict, json.load(f))
        return UserConfig(
            show_history_previews=data.get("show_history_previews", True),
            show_prompt=data.get("show_prompt", True),
            blend_falloff=data.get("blend_falloff", DEFAULT_BLEND_FALLOFF),
            click_repainting=data.get("click_repainting", True),
            recording_enabled=data.get("recording_enabled", False),
        )
    except (json.JSONDecodeError, OSError):
        return UserConfig()


def save_user_config(user_config: UserConfig) -> None:
    """Save user configuration to config_user.json."""
    config_path = _get_user_config_path()
    data: UserConfigDict = {
        "show_history_previews": user_config.show_history_previews,
        "show_prompt": user_config.show_prompt,
        "blend_falloff": user_config.blend_falloff,
        "click_repainting": user_config.click_repainting,
        "recording_enabled": user_config.recording_enabled,
    }
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
        _ = f.write("\n")
