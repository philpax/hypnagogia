"""Entry points for the client."""

import argparse
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from config import get_config
from engine import Engine

from constants import load_prompts
from game_loop import run_loop


async def main(
    *,
    comfyui_url: str,
    prompt: str,
    image_seed: int | None = None,
    n_frames: int,
    device: str,
    model: str,
    quant: str | None,
    i2i_interval: int,
    i2i_vlm_regen: bool,
    denoise: float,
    mouse_sensitivity: float,
    vision_api_url: str,
    vision_model: str,
) -> None:
    """Async main entry point."""
    config = get_config()
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

    def _cuda_warmup() -> None:
        with torch.cuda.device(device):
            _ = torch.cuda.current_blas_handle()

    await asyncio.to_thread(_cuda_warmup)

    model_config_overrides: dict[str, Any] = {"n_frames": n_frames}
    if config.models.vae_uri is not None:
        model_config_overrides["ae_uri"] = config.models.vae_uri

    engine = Engine(
        model,
        device=device,
        quant=quant,
        model_config_overrides=model_config_overrides,
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
        i2i_vlm_regen=i2i_vlm_regen,
        denoise=denoise,
        vision_api_url=vision_api_url,
        vision_model=vision_model,
    )


def cli() -> None:
    """CLI entry point with argument parsing."""
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Local World client with ComfyUI seed generation"
    )
    _ = parser.add_argument(
        "--url",
        required=True,
        help="ComfyUI server URL (e.g., http://127.0.0.1:8188)",
    )
    _ = parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt for seed image generation (default: random from prompts.txt)",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for image generation (default: random)",
    )
    _ = parser.add_argument(
        "--n-frames",
        type=int,
        default=config.defaults.n_frames,
        help=f"Number of frames (default: {config.defaults.n_frames})",
    )
    _ = parser.add_argument(
        "--i2i-interval",
        type=int,
        default=config.defaults.i2i_interval,
        help=f"Frames between i2i regeneration (default: {config.defaults.i2i_interval}, 0 to disable)",
    )
    _ = parser.add_argument(
        "--i2i-vlm-regen",
        action="store_true",
        default=config.defaults.i2i_vlm_regen,
        help="Use VLM to generate new prompt before i2i regeneration (requires --i2i-interval)",
    )
    _ = parser.add_argument(
        "--denoise",
        type=float,
        default=config.i2i.denoise,
        help=f"Denoising factor for i2i regeneration (default: {config.i2i.denoise})",
    )
    _ = parser.add_argument(
        "--model",
        default=config.models.world_engine,
        help=f"World engine model URI (default: {config.models.world_engine})",
    )
    _ = parser.add_argument(
        "--device",
        default=config.defaults.device,
        help=f"Device to use (default: {config.defaults.device})",
    )
    _ = parser.add_argument(
        "--quant",
        choices=["intw8a8", "fp8w8a8", "nvfp4"],
        default=config.defaults.quant,
        help=f"Quantization (default: {config.defaults.quant or 'none'})",
    )
    _ = parser.add_argument(
        "--mouse-sensitivity",
        type=float,
        default=config.defaults.mouse_sensitivity,
        help=f"Mouse sensitivity (default: {config.defaults.mouse_sensitivity})",
    )
    _ = parser.add_argument(
        "--vision-api-url",
        default=config.vision.api_url,
        help=f"Vision API URL (default: {config.vision.api_url})",
    )
    _ = parser.add_argument(
        "--vision-model",
        default=config.vision.model,
        help=f"Vision model name (default: {config.vision.model})",
    )
    args = parser.parse_args()

    # Pick random prompt from prompts.txt if not specified
    prompt: str | None = args.prompt  # pyright: ignore[reportAny]
    if prompt is None:
        prompts = load_prompts()
        if prompts:
            prompt = random.choice(prompts)
            print(f"Using random prompt: {prompt}")
        else:
            parser.error("--prompt is required (no prompts.txt found)")

    url: str = args.url  # pyright: ignore[reportAny]
    seed: int | None = args.seed  # pyright: ignore[reportAny]
    n_frames: int = args.n_frames  # pyright: ignore[reportAny]
    model: str = args.model  # pyright: ignore[reportAny]
    device: str = args.device  # pyright: ignore[reportAny]
    quant: str | None = args.quant  # pyright: ignore[reportAny]
    i2i_interval: int = args.i2i_interval  # pyright: ignore[reportAny]
    i2i_vlm_regen: bool = args.i2i_vlm_regen  # pyright: ignore[reportAny]
    denoise: float = args.denoise  # pyright: ignore[reportAny]
    mouse_sensitivity: float = args.mouse_sensitivity  # pyright: ignore[reportAny]
    vision_api_url: str = args.vision_api_url  # pyright: ignore[reportAny]
    vision_model: str = args.vision_model  # pyright: ignore[reportAny]

    asyncio.run(
        main(
            comfyui_url=url,
            prompt=prompt,
            image_seed=seed,
            n_frames=n_frames,
            device=device,
            model=model,
            quant=quant,
            i2i_interval=i2i_interval,
            i2i_vlm_regen=i2i_vlm_regen,
            denoise=denoise,
            mouse_sensitivity=mouse_sensitivity,
            vision_api_url=vision_api_url,
            vision_model=vision_model,
        )
    )


if __name__ == "__main__":
    cli()
