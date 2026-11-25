"""Utilities for block-wise latent refresh experiments."""

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def hybrid_refresh_ratio(
    sensitivity_error: float,
    skip_threshold: float,
    full_threshold: float,
    min_ratio: float,
) -> float:
    """Map SenCache sensitivity to the fraction of blocks to refresh.

    Errors below ``skip_threshold`` reuse the cache, errors above
    ``full_threshold`` request a dense refresh, and the middle region is
    linearly interpolated from ``min_ratio`` to one.
    """
    if skip_threshold < 0 or full_threshold <= skip_threshold:
        raise ValueError("thresholds must satisfy 0 <= skip < full")
    if not 0.0 < min_ratio <= 1.0:
        raise ValueError("min_ratio must be in (0, 1]")
    if sensitivity_error <= skip_threshold:
        return 0.0
    if sensitivity_error >= full_threshold:
        return 1.0
    position = ((sensitivity_error - skip_threshold) /
                (full_threshold - skip_threshold))
    return min_ratio + position * (1.0 - min_ratio)


def _triple(values: Sequence[int]) -> Tuple[int, int, int]:
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("block_shape must contain three positive integers")
    return int(values[0]), int(values[1]), int(values[2])


def block_refresh_masks(
    current_latent: torch.Tensor,
    cached_latent: torch.Tensor,
    token_grid: Sequence[int],
    block_shape: Sequence[int],
    refresh_ratio: float,
):
    """Rank token-grid blocks by mean squared latent change.

    Returns a residual-token mask, a latent-reference mask, and block scores.
    Latents must have shape ``[C, T, H, W]``.
    """
    if current_latent.shape != cached_latent.shape or current_latent.ndim != 4:
        raise ValueError("current_latent and cached_latent must have shape [C,T,H,W]")
    if not 0.0 <= refresh_ratio <= 1.0:
        raise ValueError("refresh_ratio must be in [0, 1]")

    grid_t, grid_h, grid_w = _triple(token_grid)
    block_t, block_h, block_w = _triple(block_shape)
    delta = (current_latent.float() - cached_latent.float()).square().mean(0)
    token_scores = F.adaptive_avg_pool3d(
        delta[None, None], (grid_t, grid_h, grid_w)
    )[0, 0]

    pad_t = (-grid_t) % block_t
    pad_h = (-grid_h) % block_h
    pad_w = (-grid_w) % block_w
    padded = F.pad(token_scores, (0, pad_w, 0, pad_h, 0, pad_t))
    valid = F.pad(torch.ones_like(token_scores), (0, pad_w, 0, pad_h, 0, pad_t))
    pool_kwargs = dict(
        kernel_size=(block_t, block_h, block_w),
        stride=(block_t, block_h, block_w),
        divisor_override=1,
    )
    score_sums = F.avg_pool3d(padded[None, None], **pool_kwargs)[0, 0]
    token_counts = F.avg_pool3d(valid[None, None], **pool_kwargs)[0, 0]
    scores = score_sums / token_counts.clamp_min(1)

    block_mask = torch.zeros_like(scores, dtype=torch.bool).flatten()
    if refresh_ratio > 0:
        order = torch.argsort(scores.flatten(), descending=True)
        ordered_counts = token_counts.flatten()[order]
        token_budget = max(1, int(round(token_scores.numel() * refresh_ratio)))
        # Select whole blocks in score order until the token budget is reached.
        selected_count = int((ordered_counts.cumsum(0) < token_budget).sum().item()) + 1
        selected_count = min(selected_count, scores.numel())
        block_mask[order[:selected_count]] = True
    block_mask = block_mask.view_as(scores)

    token_mask_3d = block_mask.repeat_interleave(block_t, 0)
    token_mask_3d = token_mask_3d.repeat_interleave(block_h, 1)
    token_mask_3d = token_mask_3d.repeat_interleave(block_w, 2)
    token_mask_3d = token_mask_3d[:grid_t, :grid_h, :grid_w]
    token_mask = token_mask_3d.reshape(1, -1, 1)

    latent_mask = F.interpolate(
        token_mask_3d[None, None].float(),
        size=current_latent.shape[1:],
        mode="nearest",
    )[0, 0].bool().unsqueeze(0).expand_as(current_latent)
    return token_mask, latent_mask, scores
