import torch

from spatial_cache import block_refresh_masks, hybrid_refresh_ratio


def test_hybrid_refresh_ratio_boundaries_and_interpolation():
    assert hybrid_refresh_ratio(4.0, 5.0, 10.0, 0.2) == 0.0
    assert hybrid_refresh_ratio(5.0, 5.0, 10.0, 0.2) == 0.0
    assert hybrid_refresh_ratio(10.0, 5.0, 10.0, 0.2) == 1.0
    assert hybrid_refresh_ratio(12.0, 5.0, 10.0, 0.2) == 1.0
    assert abs(hybrid_refresh_ratio(7.5, 5.0, 10.0, 0.2) - 0.6) < 1e-8


def test_top_changed_block_is_selected():
    cached = torch.zeros(4, 2, 8, 8)
    current = cached.clone()
    current[:, :, 4:, 4:] = 10
    token_mask, latent_mask, scores = block_refresh_masks(
        current, cached, token_grid=(2, 8, 8), block_shape=(2, 4, 4), refresh_ratio=0.25
    )
    assert scores.numel() == 4
    assert token_mask.sum().item() == 32
    assert latent_mask[:, :, 4:, 4:].all()
    assert not latent_mask[:, :, :4, :4].any()


def test_zero_and_full_refresh():
    current = torch.randn(3, 3, 7, 9)
    cached = torch.zeros_like(current)
    zero, _, _ = block_refresh_masks(current, cached, (3, 7, 9), (1, 4, 4), 0.0)
    full, _, _ = block_refresh_masks(current, cached, (3, 7, 9), (1, 4, 4), 1.0)
    assert not zero.any()
    assert full.all()


def test_irregular_edges_follow_token_budget():
    current = torch.randn(4, 3, 7, 9)
    cached = torch.zeros_like(current)
    mask, _, _ = block_refresh_masks(current, cached, (3, 7, 9), (1, 4, 4), 0.25)
    target = round(3 * 7 * 9 * 0.25)
    assert target <= mask.sum().item() < target + 16
