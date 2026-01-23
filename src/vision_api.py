"""Vision API client for scene description."""

import base64
import io
import os
from dataclasses import dataclass
from typing import TypedDict

import requests
import torch
from PIL import Image


class _MessageDict(TypedDict):
    content: str


class _ChoiceDict(TypedDict):
    message: _MessageDict


class _APIResponseDict(TypedDict):
    choices: list[_ChoiceDict]


VISION_SYSTEM_PROMPT = """You are a scene descriptor. Analyze the image and describe it as an image generation prompt.

Format rules:
- Start with "First-person view, "
- Identify the art style (e.g. oil painting, watercolour, pixel art, cel-shaded, photorealistic, impressionist, surrealist, art nouveau, cyberpunk, vaporwave)
- Describe the landscape, time of day, and atmosphere
- End with "sharp stable framing"
- Keep under 50 words total
- No text, brands, people, or body parts in descriptions

Examples:
- First-person view, impressionist oil painting style, misty marsh landscape at dawn, thatched cottages rising from reeds, golden light filtering through fog, soft visible brushstrokes, sharp stable framing
- First-person view, vaporwave aesthetic, vast neon-lit cityscape under starry night sky, dramatic skyscraper silhouettes, vibrant turquoise and magenta palette, soft glowing haze, sharp stable framing
- First-person view, Studio Ghibli style, ancient redwood forest at twilight, towering moss-covered trunks, soft golden rays piercing through canopy, whimsical atmosphere, sharp stable framing"""


@dataclass
class VisionResult:
    success: bool
    prompt: str | None
    error: str | None


def _tensor_to_base64_jpeg(tensor: torch.Tensor) -> str:
    """Convert a frame tensor (H,W,3) uint8 to base64 JPEG."""
    if tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 255).to(torch.uint8)

    np_image = tensor.cpu().numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    image = Image.fromarray(np_image, mode="RGB")  # pyright: ignore[reportUnknownArgumentType]

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    _ = buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def describe_frame(
    frame: torch.Tensor,
    api_url: str,
    model: str,
    api_key_env: str,
    max_tokens: int,
    timeout: float,
) -> VisionResult:
    """Send frame to vision API and get a scene description prompt.

    Args:
        frame: Image tensor (H,W,3) uint8 or float
        api_url: OpenAI-compatible API base URL
        model: Model name (e.g., "gpt-4o")
        api_key_env: Environment variable name containing API key
        max_tokens: Maximum tokens for response
        timeout: Request timeout in seconds

    Returns:
        VisionResult with success status and prompt or error
    """
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        return VisionResult(
            success=False,
            prompt=None,
            error=f"API key not found in environment variable: {api_key_env}",
        )

    try:
        base64_image = _tensor_to_base64_jpeg(frame)
    except Exception as e:
        return VisionResult(
            success=False,
            prompt=None,
            error=f"Failed to encode image: {e}",
        )

    url = f"{api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this scene as an image generation prompt.",
                    },
                ],
            },
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        return VisionResult(
            success=False,
            prompt=None,
            error=f"Request timed out after {timeout}s",
        )
    except requests.RequestException as e:
        return VisionResult(
            success=False,
            prompt=None,
            error=f"API request failed: {e}",
        )

    try:
        data: _APIResponseDict = response.json()  # pyright: ignore[reportAny]
        prompt_text = data["choices"][0]["message"]["content"].strip()
        return VisionResult(success=True, prompt=prompt_text, error=None)
    except (KeyError, IndexError, ValueError) as e:
        return VisionResult(
            success=False,
            prompt=None,
            error=f"Failed to parse API response: {e}",
        )
