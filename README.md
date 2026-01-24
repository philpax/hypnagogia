# Hypnagogia

An interactive dream machine, forked from [Overworld's local gameplay client](https://github.com/Overworldai/local_world), with ComfyUI integration.

<video src="video.mp4" controls muted autoplay loop></video>

The first frame is generated using text-to-image (Z Image Turbo). You can then explore the dream world using WASD + mouse, with control over its direction:

- **Left-click** regenerates the current frame using image-to-image (SDXL Turbo) and injects the result back into the world model, partially restabilising the dream while converting local hallucinations into global ones
- **Right-click** analyses the current frame with a vision language model (e.g. GPT-4o) to generate a new prompt based on what it sees, letting the dream evolve organically
- **Scroll wheel** adjusts the denoise strength for I2I regeneration (0-100%)
- **ESC** opens a pause menu for editing the prompt, selecting from presets, or adjusting settings

Periodic automatic repainting is supported via the `--i2i-interval` flag, but is off by default.

## Requirements

For the world model:

- Windows
- Nvidia RTX 3xxx / 4xxx / 5xxx GPU
- CUDA 12.8

For the generations:

- ComfyUI (on another GPU/machine; running both models on the same GPU will likely make you very sad)
- Hardware capable of generating SDXL Turbo outputs in <100ms (faster the better)

For vision analysis (optional):

- OpenAI API key (or compatible vision API endpoint)

## Configuration

Update `config.json` as appropriate. The ComfyUI workflows are specialised to the default models, so changing them is unlikely to work out of the box.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | (required) | ComfyUI server URL |
| `--prompt` | random from prompts.txt | Initial text prompt |
| `--seed` | random | Seed for reproducible generation |
| `--n-frames` | 4096 | Frame buffer size before auto-reset |
| `--i2i-interval` | 0 (disabled) | Frames between automatic I2I regenerations |
| `--device` | cuda | PyTorch device |
| `--mouse-sensitivity` | 1.5 | Mouse control multiplier |
| `--vision-api-url` | https://api.openai.com/v1 | Vision API base URL |
| `--vision-model` | gpt-4o | Vision model name |

### Example

```
uv run src/client.py --url http://comfyui:8188 --vision-api-url http://vlm-server:7070/v1 --vision-model gpu:qwen3-vl-30b-a3b-instruct
```

## Controls

| Input | Action |
|-------|--------|
| WASD + Mouse | Navigate the dream world |
| Left-click | Regenerate frame with I2I and inject |
| Right-click | Analyse frame with VLM for new prompt |
| Scroll wheel | Adjust denoise strength |
| ESC | Open pause menu |
| U | Manual reset |
