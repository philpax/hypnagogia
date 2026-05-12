"""WorldEngine wrapper exposing temporal compression + inference FPS.

The wrapper hides multi-frame batching: callers pass single ``(H, W, 3)``
seed/injection frames and the wrapper replicates them across the temporal
dimension when the underlying model expects ``(tc, H, W, 3)``.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from world_engine import CtrlInput, WorldEngine

# Fraction of a sleep interval yielded to the OS via ``pygame.time.wait``;
# the remainder is covered by a busy-wait spin for precise timing. OS
# schedulers may extend sleeps beyond their stated duration, so we
# deliberately undershoot and spin the rest.
SLEEP_RATIO = 0.8


class Engine:
    """Thin wrapper over :class:`world_engine.WorldEngine`.

    Acts as a drop-in replacement for the bare ``WorldEngine`` in hypnagogia's
    game loop, with the added behaviour that ``append_frame`` automatically
    expands a single ``(H, W, 3)`` tensor to ``(tc, H, W, 3)`` for models that
    consume temporally-compressed frame stacks (Waypoint-1.5 and later).
    """

    def __init__(
        self,
        model_uri: str,
        *,
        device: str,
        quant: str | None = None,
        model_config_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.model_uri: str = model_uri
        self.inner: WorldEngine = WorldEngine(
            model_uri,
            quant=quant,
            device=device,
            model_config_overrides=model_config_overrides,
        )

    @property
    def temporal_compression(self) -> int:
        return cast(int, getattr(self.inner.model_cfg, "temporal_compression", 1))

    @property
    def inference_fps(self) -> int:
        return cast(int, getattr(self.inner.model_cfg, "inference_fps", 30))

    @property
    def device(self) -> str:
        return cast(str, self.inner.device)

    def _expand(self, frame: torch.Tensor) -> torch.Tensor:
        """Replicate a ``(H, W, 3)`` frame across the temporal dim if tc > 1."""
        tc = self.temporal_compression
        if tc > 1 and frame.dim() == 3:
            return frame.unsqueeze(0).expand(tc, -1, -1, -1).contiguous()
        return frame

    def reset(self) -> None:
        self.inner.reset()

    def append_frame(
        self, frame: torch.Tensor, ctrl: CtrlInput | None = None
    ) -> torch.Tensor | None:
        if ctrl is None:
            return self.inner.append_frame(self._expand(frame))
        return self.inner.append_frame(self._expand(frame), ctrl=ctrl)

    def gen_frame(self, *, ctrl: CtrlInput) -> torch.Tensor:
        return self.inner.gen_frame(ctrl=ctrl)

    def warmup(self) -> torch.Tensor:
        """Run one ``gen_frame`` to trigger ``torch.compile``.

        Returns the first frame on CPU so the caller can use it as the
        initial ``pending`` batch when entering the pipeline loop.
        """
        return self.inner.gen_frame(ctrl=CtrlInput()).cpu()
