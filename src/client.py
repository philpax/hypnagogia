import argparse
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import AsyncIterator

import pygame
import torch
from world_engine import CtrlInput, WorldEngine

from seed_gen import generate_i2i, generate_t2i

# Separate executor for i2i so it doesn't block the engine
_i2i_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="i2i")

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


async def ctrl_stream(
    restart_event: asyncio.Event,
    pause_event: asyncio.Event,
    mouse_sensitivity: float = 1.5,
    whitelisted_keys=None,
) -> AsyncIterator[CtrlInput]:
    whitelisted_keys = WHITELIST_KEYS if whitelisted_keys is None else whitelisted_keys

    codes = (
        {("k", k): v for k, v in PYGAME_TO_VK.items()}
        | {
            ("m", 1): 0x01,
            ("m", 2): 0x04,
            ("m", 3): 0x02,
        }  # note: pygame has middle wheel as m2
    )
    codes = {k: v for k, v in codes.items() if v in whitelisted_keys}

    held: set[int] = set()

    while True:
        btn: set[int] = set()

        for e in pygame.event.get():  # edge presses + drain
            if e.type == pygame.QUIT:
                return

            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pause_event.set()
                elif e.key == pygame.K_u:
                    restart_event.set()

                c = codes.get(("k", e.key))
                if c is not None:
                    btn.add(c)
                    held.add(c)

            elif e.type == pygame.KEYUP:
                c = codes.get(("k", e.key))
                if c is not None:
                    held.discard(c)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                c = codes.get(("m", e.button))
                if c is not None:
                    btn.add(c)

        btn.update(held)

        mb = pygame.mouse.get_pressed(3)
        btn.update(
            c
            for i, down in enumerate(mb, 1)
            if down and (c := codes.get(("m", i))) is not None
        )

        dx, dy = pygame.mouse.get_rel()
        yield CtrlInput(
            button=btn, mouse=(dx * mouse_sensitivity, dy * mouse_sensitivity)
        )
        await asyncio.sleep(0)


async def run_loop(
    *,
    engine: WorldEngine,
    seed_frame: torch.Tensor | None,
    n_frames: int,
    mouse_sensitivity: float = 1.5,
    comfyui_url: str | None = None,
    prompt: str | None = None,
    image_seed: int | None = None,
    i2i_interval: int = 120,
) -> None:
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    pygame.display.set_caption("U=restart, ESC=pause, close window to exit")

    try:
        pygame.event.set_grab(True)

        restart = asyncio.Event()
        pause = asyncio.Event()
        ctrls = ctrl_stream(
            restart_event=restart,
            pause_event=pause,
            mouse_sensitivity=mouse_sensitivity,
        )
        limit = max(1, n_frames - 2)

        async def reset(*, reload_seed: bool = False) -> None:
            nonlocal seed_frame
            await asyncio.to_thread(engine.reset)
            if reload_seed or seed_frame is None:
                if comfyui_url and prompt:
                    seed_frame = await asyncio.to_thread(
                        generate_t2i, comfyui_url, prompt, image_seed
                    )
                else:
                    raise ValueError(
                        "ComfyUI URL and prompt are required for seed generation"
                    )
            if seed_frame is not None:
                await asyncio.to_thread(engine.append_frame, seed_frame)

        i2i_future: Future | None = None

        def draw(img: torch.Tensor) -> None:
            img = img.detach()
            if img.dtype != torch.uint8:
                img = img.clamp(0, 255).to(torch.uint8)
            frame = img.cpu().numpy()  # (H,W,3)
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # (W,H,3)
            surf = pygame.transform.scale(surf, screen.get_size())
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        async def show_pause_menu() -> bool:
            """Show pause menu. Returns True to resume, False to quit."""
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            font = pygame.font.SysFont(None, 48)
            small_font = pygame.font.SysFont(None, 32)

            # Capture current frame as background
            background = screen.copy()

            resume_rect = pygame.Rect(0, 0, 200, 50)
            quit_rect = pygame.Rect(0, 0, 200, 50)

            while True:
                sw, sh = screen.get_size()
                resume_rect.center = (sw // 2, sh // 2 - 30)
                quit_rect.center = (sw // 2, sh // 2 + 40)

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return False
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_ESCAPE:
                            return True
                    if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                        if resume_rect.collidepoint(e.pos):
                            return True
                        if quit_rect.collidepoint(e.pos):
                            return False

                # Redraw background each frame
                screen.blit(background, (0, 0))

                # Draw transparent overlay
                overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 100))
                screen.blit(overlay, (0, 0))

                # Draw title
                title = font.render("PAUSED", True, (255, 255, 255))
                screen.blit(title, (sw // 2 - title.get_width() // 2, sh // 2 - 120))

                # Draw buttons
                mouse_pos = pygame.mouse.get_pos()
                for rect, text in [(resume_rect, "Resume"), (quit_rect, "Quit")]:
                    color = (
                        (100, 100, 100, 200)
                        if rect.collidepoint(mouse_pos)
                        else (60, 60, 60, 200)
                    )
                    btn_surf = pygame.Surface(
                        (rect.width, rect.height), pygame.SRCALPHA
                    )
                    pygame.draw.rect(
                        btn_surf, color, btn_surf.get_rect(), border_radius=8
                    )
                    pygame.draw.rect(
                        btn_surf,
                        (255, 255, 255),
                        btn_surf.get_rect(),
                        2,
                        border_radius=8,
                    )
                    screen.blit(btn_surf, rect.topleft)
                    label = small_font.render(text, True, (255, 255, 255))
                    screen.blit(
                        label,
                        (
                            rect.centerx - label.get_width() // 2,
                            rect.centery - label.get_height() // 2,
                        ),
                    )

                pygame.display.flip()
                await asyncio.sleep(1 / 60)

        await reset(reload_seed=True)

        frames = 0
        last_frame: torch.Tensor | None = None
        async for ctrl in ctrls:
            # Check if i2i task completed (non-blocking)
            if i2i_future is not None and i2i_future.done():
                refreshed = i2i_future.result()
                await asyncio.to_thread(engine.append_frame, refreshed)
                i2i_future = None

            if pause.is_set():
                pause.clear()
                if not await show_pause_menu():
                    return
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                pygame.mouse.get_rel()  # discard accumulated mouse movement
                continue

            if restart.is_set() or frames >= limit:
                restart.clear()
                await reset(reload_seed=False)
                frames = 0

            img = await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
            frames += 1
            last_frame = img
            draw(img)

            # Start i2i regeneration every i2i_interval frames (if not already running)
            if (
                i2i_interval > 0
                and frames > 0
                and frames % i2i_interval == 0
                and i2i_future is None
                and comfyui_url
                and prompt
            ):
                i2i_future = _i2i_executor.submit(
                    generate_i2i, comfyui_url, prompt, last_frame
                )

            await asyncio.sleep(0)
    finally:
        pygame.event.set_grab(False)
        pygame.quit()


async def main(
    *,
    comfyui_url: str,
    prompt: str,
    image_seed: int | None = None,
    n_frames: int = 4096,
    device: str = "cuda",
    i2i_interval: int = 120,
) -> None:
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

    def _cuda_warmup() -> None:
        with torch.cuda.device(device):
            torch.cuda.current_blas_handle()

    await asyncio.to_thread(_cuda_warmup)

    engine = WorldEngine(
        "Overworld/Waypoint-1-Small",
        device=device,
        model_config_overrides={
            "n_frames": n_frames,
            "ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan",
        },
    )
    await run_loop(
        engine=engine,
        seed_frame=None,
        n_frames=n_frames,
        mouse_sensitivity=1.5,
        comfyui_url=comfyui_url,
        prompt=prompt,
        image_seed=image_seed,
        i2i_interval=i2i_interval,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Local World client with ComfyUI seed generation"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="ComfyUI server URL (e.g., http://127.0.0.1:8188)",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for seed image generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for image generation (default: random)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=4096,
        help="Number of frames (default: 4096)",
    )
    parser.add_argument(
        "--i2i-interval",
        type=int,
        default=120,
        help="Frames between i2i regeneration (default: 120, 0 to disable)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            comfyui_url=args.url,
            prompt=args.prompt,
            image_seed=args.seed,
            n_frames=args.n_frames,
            device=args.device,
            i2i_interval=args.i2i_interval,
        )
    )


if __name__ == "__main__":
    cli()
