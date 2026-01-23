"""Configuration loader for local_world."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


class WindowConfigDict(TypedDict):
    width: int
    height: int


class DefaultsConfigDict(TypedDict):
    n_frames: int
    i2i_interval: int
    device: str
    mouse_sensitivity: float


class ModelsConfigDict(TypedDict):
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
    device: str
    mouse_sensitivity: float


@dataclass
class ModelsConfig:
    world_engine: str
    vae_uri: str


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
