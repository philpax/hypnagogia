"""Generate seed images using ComfyScript with the z-image-turbo workflow."""

import random

import numpy as np
import torch
import torch.nn.functional as F


def generate_seed_image(
    comfyui_url: str,
    prompt: str,
    seed: int | None = None,
    target_size: tuple[int, int] = (360, 640),
) -> torch.Tensor:
    """Generate seed image using ComfyScript with the z-image-turbo workflow."""
    if seed is None:
        seed = random.randint(0, 2**53 - 1)

    from comfy_script.runtime import Workflow, load, util
    load(comfyui_url)

    from comfy_script.runtime.nodes import (
        CLIPLoader,
        CLIPTextEncode,
        ConditioningZeroOut,
        EmptySD3LatentImage,
        KSampler,
        ModelSamplingAuraFlow,
        UNETLoader,
        VAEDecode,
        VAELoader,
    )

    with Workflow(wait=True):
        clip = CLIPLoader("qwen_3_4b.safetensors", "lumina2", "default")
        vae = VAELoader("flux1_ae.safetensors")
        latent = EmptySD3LatentImage(640, 368, 1)
        positive = CLIPTextEncode(prompt, clip)
        negative = ConditioningZeroOut(positive)
        unet = UNETLoader("z_image_turbo_bf16.safetensors", "default")
        model = ModelSamplingAuraFlow(unet, 3)
        samples = KSampler(
            model, seed, 9, 1, "res_multistep", "simple", positive, negative, latent, 1
        )
        image = VAEDecode(samples, vae)
        images = util.get_images(image)

    # Convert PIL image to tensor
    pil_img = images[0].convert("RGB")
    img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).float()
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]
    return frame.to(dtype=torch.uint8).permute(1, 2, 0).contiguous()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate seed image using ComfyScript")
    parser.add_argument("--url", required=True, help="ComfyUI server URL")
    parser.add_argument("--prompt", default="First-person view, sun-drenched rocky desert path winding through jagged cliffs, warm golden light casting long shadows, distant mesas under a hazy blue sky", help="Text prompt")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation")
    parser.add_argument("--output", default="seed_test.png", help="Output filename")
    args = parser.parse_args()

    print(f"Connecting to {args.url}...")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed or 'random'}")

    tensor = generate_seed_image(args.url, args.prompt, args.seed)
    print(f"Generated tensor: {tensor.shape}, dtype={tensor.dtype}")

    # Save as image
    from PIL import Image
    img = Image.fromarray(tensor.numpy())
    img.save(args.output)
    print(f"Saved to {args.output}")
