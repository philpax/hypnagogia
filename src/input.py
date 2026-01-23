"""Input handling for the client."""

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import pygame
from world_engine import CtrlInput

from constants import HISTORY_BROWSE_KEY, PYGAME_TO_VK, WHITELIST_KEYS
from state import ClientState, GameState


async def ctrl_stream(
    restart_event: asyncio.Event,
    pause_event: asyncio.Event,
    mouse_sensitivity: float,
    state: ClientState,
    on_scroll: Callable[[int], None] | None = None,
    on_history_click: Callable[[tuple[int, int]], None] | None = None,
    whitelisted_keys: frozenset[int] | None = None,
) -> AsyncIterator[CtrlInput]:
    """Async generator that yields control inputs from pygame events."""
    if whitelisted_keys is None:
        whitelisted_keys = WHITELIST_KEYS

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
                if e.key == pygame.K_ESCAPE:  # pyright: ignore[reportAny]
                    # Accumulate play time before pausing
                    if state.play_start is not None:
                        state.play_time += time.time() - state.play_start
                        state.play_start = None
                    state.game_state = GameState.PAUSED
                    state.apply_game_state()
                    pause_event.set()
                elif e.key == pygame.K_u:  # pyright: ignore[reportAny]
                    restart_event.set()
                elif e.key == HISTORY_BROWSE_KEY:  # pyright: ignore[reportAny]
                    state.game_state = GameState.BROWSING
                    state.apply_game_state()
                    continue  # Don't forward Q to game

                c = codes.get(("k", e.key))  # pyright: ignore[reportAny]
                if c is not None:
                    btn.add(c)
                    held.add(c)

            elif e.type == pygame.KEYUP:
                if e.key == HISTORY_BROWSE_KEY:  # pyright: ignore[reportAny]
                    state.game_state = GameState.PLAYING
                    state.apply_game_state()
                    _ = pygame.mouse.get_rel()  # Discard accumulated mouse movement
                    continue  # Don't forward Q to game

                c = codes.get(("k", e.key))  # pyright: ignore[reportAny]
                if c is not None:
                    held.discard(c)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                # When browsing, LMB clicks go to history instead of game
                browsing = state.game_state == GameState.BROWSING
                if browsing and e.button == 1 and on_history_click:  # pyright: ignore[reportAny]
                    on_history_click(e.pos)  # pyright: ignore[reportAny]
                else:
                    c = codes.get(("m", e.button))
                    if c is not None:
                        btn.add(c)

            elif e.type == pygame.MOUSEWHEEL and on_scroll is not None:
                on_scroll(e.y)  # pyright: ignore[reportAny]

        btn.update(held)

        # When browsing, don't pass mouse buttons to game
        browsing = state.game_state == GameState.BROWSING
        if not browsing:
            mb = pygame.mouse.get_pressed(3)
            btn.update(
                c
                for i, down in enumerate(mb, 1)
                if down and (c := codes.get(("m", i))) is not None
            )

        dx, dy = pygame.mouse.get_rel()
        # When browsing or paused, don't pass mouse movement to game
        if state.game_state != GameState.PLAYING:
            dx, dy = 0, 0
        else:
            # Center cursor when playing
            center = (state.screen.get_width() // 2, state.screen.get_height() // 2)
            pygame.mouse.set_pos(center)
        yield CtrlInput(
            button=btn, mouse=(dx * mouse_sensitivity, dy * mouse_sensitivity)
        )
        await asyncio.sleep(0)
