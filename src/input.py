"""Input handling for the client."""

import asyncio
from collections.abc import Callable
from typing import AsyncIterator

import pygame
from world_engine import CtrlInput

from constants import HISTORY_BROWSE_KEY, PYGAME_TO_VK, WHITELIST_KEYS


async def ctrl_stream(
    restart_event: asyncio.Event,
    pause_event: asyncio.Event,
    mouse_sensitivity: float,
    on_scroll: Callable[[int], None] | None = None,
    on_browse_change: Callable[[bool], None] | None = None,
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
    browsing: bool = False

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
                elif e.key == HISTORY_BROWSE_KEY:
                    browsing = True
                    if on_browse_change:
                        on_browse_change(True)
                    continue  # Don't forward Q to game

                c = codes.get(("k", e.key))
                if c is not None:
                    btn.add(c)
                    held.add(c)

            elif e.type == pygame.KEYUP:
                if e.key == HISTORY_BROWSE_KEY:
                    browsing = False
                    if on_browse_change:
                        on_browse_change(False)
                    continue  # Don't forward Q to game

                c = codes.get(("k", e.key))
                if c is not None:
                    held.discard(c)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                # When browsing, LMB clicks go to history instead of game
                if browsing and e.button == 1 and on_history_click:
                    on_history_click(e.pos)
                else:
                    c = codes.get(("m", e.button))
                    if c is not None:
                        btn.add(c)

            elif e.type == pygame.MOUSEWHEEL and on_scroll is not None:
                on_scroll(e.y)  # e.y is positive for scroll up, negative for down

        btn.update(held)

        # When browsing, don't pass mouse buttons to game
        if not browsing:
            mb = pygame.mouse.get_pressed(3)
            btn.update(
                c
                for i, down in enumerate(mb, 1)
                if down and (c := codes.get(("m", i))) is not None
            )

        dx, dy = pygame.mouse.get_rel()
        # When browsing, don't pass mouse movement to game
        if browsing:
            dx, dy = 0, 0
        yield CtrlInput(
            button=btn, mouse=(dx * mouse_sensitivity, dy * mouse_sensitivity)
        )
        await asyncio.sleep(0)
