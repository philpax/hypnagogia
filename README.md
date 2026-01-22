# Hypnagogia

A fork of [Overworld's local gameplay client](https://github.com/Overworldai/local_world) with ComfyUI integration.

The first frame is generated using text-to-image (currently Z Image Turbo). The most recent frame is periodically repainted with image-to-image (currently SDXL Turbo); the result is then injected back into the model to partially restabilise the dream, simultaneously converting local hallucinations to global hallucinations in the process.

## Requirements

For the world model:

- Windows
- Nvidia RTX 3xxx / 4xxx / 5xxx GPU
- CUDA 12.8

For the generations:

- ComfyUI (on another GPU/machine; running both models on the same GPU will likely make you very sad)
- Hardware capable of generating SDXL Turbo outputs in <100ms (faster the better)

## Settings

Update `config.json` as appropriate. Note that the ComfyUI workflows are specialised to these two models, so it's unlikely changing them will work out of the box.

Overrides can be provided for some of the settings in the CLI arguments:
```
uv run src/client.py --url http://comfyui:8188 --prompt "First-person view, sun-drenched rocky desert path winding through jagged cliffs, warm golden light casting long shadows, distant mesas under a hazy blue sky, sharp stable framing"
```
