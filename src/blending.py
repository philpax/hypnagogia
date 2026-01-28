"""Frame blending utilities with gradient masks."""

import torch


def create_blend_mask(height: int, width: int, falloff: float = 0.8) -> torch.Tensor:
    """Create a blend mask using Chebyshev distance.

    The mask is 1.0 at the center and falls off toward the edges based on the
    maximum of normalized x/y distances (Chebyshev distance).

    Args:
        height: Image height
        width: Image width
        falloff: Threshold where mask starts falling off (0.0 to 1.0).
                 - falloff=0.0: pure gradient from center (mask=1) to edges (mask=0)
                 - falloff=0.5: inner 50% is solid i2i, then linear gradient
                 - falloff=1.0: all i2i everywhere

    Returns:
        Tensor of shape (H, W) with values in [0, 1]
    """
    # Center of the image
    cx, cy = width / 2, height / 2

    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Chebyshev distance: max of normalized x/y distances
    # d = 0 at center, d = 1 at edge centers and corners
    d = torch.max(torch.abs(xx - cx) / (width / 2), torch.abs(yy - cy) / (height / 2))

    # Mask formula based on falloff threshold
    if falloff >= 1.0:
        mask = torch.ones_like(d)  # All i2i
    else:
        mask = ((1.0 - d) / (1.0 - falloff)).clamp(0.0, 1.0)

    return mask


def blend_frames(
    i2i_frame: torch.Tensor,
    world_frame: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend i2i frame with world model frame using blend mask.

    Args:
        i2i_frame: The image-to-image generated frame (H, W, 3)
        world_frame: The world model generated frame (H, W, 3)
        mask: Optional pre-computed mask. If None, creates one based on frame size.

    Returns:
        Blended frame tensor (H, W, 3)
    """
    h, w = i2i_frame.shape[:2]

    if mask is None:
        mask = create_blend_mask(h, w)

    # Move all tensors to same device as i2i_frame
    device = i2i_frame.device
    mask = mask.to(device)
    world_frame = world_frame.to(device)

    # Expand mask for broadcasting: (H, W) -> (H, W, 1)
    mask_3d = mask.unsqueeze(-1)

    # Blend: result = i2i * mask + world * (1 - mask)
    # Center (mask=1) = i2i, edges (mask=0) = world model
    blended = i2i_frame.float() * mask_3d + world_frame.float() * (1.0 - mask_3d)

    return blended.to(i2i_frame.dtype)
