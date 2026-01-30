"""Generate seed images using ComfyScript with the z-image-turbo workflow."""

# pyright: reportUnusedCallResult=none

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import get_config

# Engine resolution (width, height) - hardcoded constant
ENGINE_RESOLUTION = (640, 360)

# Cache for loaded state
_comfy_loaded = False


def _ensure_loaded(comfyui_url: str):
    """Load ComfyUI connection once."""
    global _comfy_loaded
    if not _comfy_loaded:
        from comfy_script.runtime import load  # pyright: ignore[reportUnknownVariableType]

        load(comfyui_url)
        _comfy_loaded = True


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (H, W, 3) uint8 tensor to PIL Image."""
    return Image.fromarray(tensor.cpu().numpy())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


def pil_to_tensor(pil_img: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
    """Convert PIL Image to (H, W, 3) uint8 tensor with resizing."""
    img = (
        torch.from_numpy(np.array(pil_img.convert("RGB")))  # pyright: ignore[reportUnknownMemberType]
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[
        0
    ]
    return frame.to(dtype=torch.uint8).permute(1, 2, 0).contiguous()


def generate_t2i(
    comfyui_url: str,
    prompt: str,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate image from text using the z-image-turbo workflow."""
    config = get_config()
    target_size = (ENGINE_RESOLUTION[1], ENGINE_RESOLUTION[0])  # (H, W)

    if seed is None:
        seed = random.randint(0, 2**53 - 1)

    _ensure_loaded(comfyui_url)

    from comfy_script.runtime import Workflow, util
    from comfy_script.runtime.nodes import (
        CLIPLoader,
        CLIPTextEncode,
        ConditioningZeroOut,
        EmptySD3LatentImage,
        KSampler,
        ModelSamplingAuraFlow,
        PreviewImage,
        UNETLoader,
        VAEDecode,
        VAELoader,
    )

    with Workflow(wait=True):
        clip = CLIPLoader(config.t2i.clip, "lumina2", "default")
        vae = VAELoader(config.t2i.vae)
        latent = EmptySD3LatentImage(ENGINE_RESOLUTION[0], ENGINE_RESOLUTION[1], 1)
        positive = CLIPTextEncode(prompt, clip)
        negative = ConditioningZeroOut(positive)
        unet = UNETLoader(config.t2i.unet, "default")
        model = ModelSamplingAuraFlow(unet, 3)
        samples = KSampler(
            model,
            seed,
            config.t2i.steps,
            config.t2i.cfg,
            config.t2i.sampler,
            "simple",
            positive,
            negative,
            latent,
            1,
        )
        image = VAEDecode(samples, vae)
        PreviewImage(image)
        images = util.get_images(image)  # pyright: ignore[reportUnknownMemberType]

    first_image = images[0]
    assert first_image is not None
    return pil_to_tensor(first_image, target_size)


def _upload_image(
    comfyui_url: str, pil_image: Image.Image, filename: str = "frame.png"
) -> str:
    """Upload an image to ComfyUI and return the filename."""
    import io
    import urllib.request

    # Convert PIL image to bytes
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Create multipart form data
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode()
        + img_bytes.read()
        + f"\r\n--{boundary}--\r\n".encode()
    )

    url = f"{comfyui_url.rstrip('/')}/upload/image"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:  # pyright: ignore[reportAny]
        import json

        result = json.loads(response.read())  # pyright: ignore[reportAny]
        return result.get("name", filename)  # pyright: ignore[reportAny]


def generate_i2i(
    comfyui_url: str,
    prompt: str,
    input_image: torch.Tensor,
    seed: int | None = None,
    denoise: float | None = None,
) -> torch.Tensor:
    """Generate image from image using the SDXL Turbo i2i workflow."""
    config = get_config()
    target_size = (ENGINE_RESOLUTION[1], ENGINE_RESOLUTION[0])  # (H, W)

    if seed is None:
        seed = random.randint(0, 2**53 - 1)
    if denoise is None:
        denoise = config.i2i.denoise

    _ensure_loaded(comfyui_url)

    from comfy_script.runtime import Workflow, util
    from comfy_script.runtime.nodes import (
        CheckpointLoaderSimple,
        CLIPTextEncode,
        ImageScale,
        KSamplerSelect,
        LoadImage,
        PreviewImage,
        SamplerCustom,
        SDTurboScheduler,
        VAEDecode,
        VAEEncode,
    )

    # Upload input image to ComfyUI
    pil_input = _tensor_to_pil(input_image)
    uploaded_name = _upload_image(comfyui_url, pil_input, "frame.png")

    with Workflow(wait=True):
        model, clip, vae = CheckpointLoaderSimple(config.i2i.checkpoint)

        # Load and encode input image
        input_img, _ = LoadImage(uploaded_name)
        scaled = ImageScale(
            input_img, "bicubic", ENGINE_RESOLUTION[0], ENGINE_RESOLUTION[1], "disabled"
        )
        latent = VAEEncode(scaled, vae)

        positive = CLIPTextEncode(prompt, clip)
        negative = CLIPTextEncode("text, watermark", clip)

        sampler = KSamplerSelect(config.i2i.sampler)
        sigmas = SDTurboScheduler(model, config.i2i.steps, denoise)

        samples, _ = SamplerCustom(
            model, True, seed, 1, positive, negative, sampler, sigmas, latent
        )
        image = VAEDecode(samples, vae)
        PreviewImage(image)
        images = util.get_images(image)  # pyright: ignore[reportUnknownMemberType]

    first_image = images[0]
    assert first_image is not None
    return pil_to_tensor(first_image, target_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate seed image using ComfyScript"
    )
    parser.add_argument("--url", required=True, help="ComfyUI server URL")
    parser.add_argument(
        "--prompt",
        default="First-person view, sun-drenched rocky desert path winding through jagged cliffs, warm golden light casting long shadows, distant mesas under a hazy blue sky",
        help="Text prompt",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation")
    parser.add_argument("--output", default="seed_test.png", help="Output filename")
    parser.add_argument(
        "--test-i2i", action="store_true", help="Also test i2i after t2i"
    )
    args = parser.parse_args()

    url: str = args.url  # pyright: ignore[reportAny]
    prompt: str = args.prompt  # pyright: ignore[reportAny]
    seed: int | None = args.seed  # pyright: ignore[reportAny]
    output: str = args.output  # pyright: ignore[reportAny]
    test_i2i: bool = args.test_i2i  # pyright: ignore[reportAny]

    print(f"Connecting to {url}...")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed or 'random'}")

    print("\n=== Testing t2i ===")
    tensor = generate_t2i(url, prompt, seed)
    print(f"Generated tensor: {tensor.shape}, dtype={tensor.dtype}")

    img = Image.fromarray(tensor.numpy())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    img.save(output)
    print(f"Saved to {output}")

    if test_i2i:
        print("\n=== Testing i2i ===")
        i2i_output = output.replace(".png", "_i2i.png")
        tensor_i2i = generate_i2i(url, prompt, tensor, seed)
        print(f"Generated i2i tensor: {tensor_i2i.shape}, dtype={tensor_i2i.dtype}")

        img_i2i = Image.fromarray(tensor_i2i.numpy())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        img_i2i.save(i2i_output)
        print(f"Saved to {i2i_output}")
