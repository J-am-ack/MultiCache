import torch

from partial_query import partial_query_block
from wan.modules.model import WanAttentionBlock, rope_params


@torch.inference_mode()
def test_full_mask_matches_dense_block():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(7)
    device = torch.device("cuda")
    dim, heads = 48, 3
    block = WanAttentionBlock(
        "t2v_cross_attn", dim, 96, heads, qk_norm=True
    ).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 16, dim, device=device, dtype=torch.bfloat16)
    e = torch.randn(1, 6, dim, device=device, dtype=torch.float32)
    context = torch.randn(1, 8, dim, device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([16], dtype=torch.long)
    grid_sizes = torch.tensor([[1, 4, 4]], dtype=torch.long)
    freqs = rope_params(16, dim // heads).to(device)
    kwargs = dict(
        e=e,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_lens=None,
    )
    mask = torch.ones(1, 16, 1, dtype=torch.bool, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense = block(x.clone(), **kwargs)
        indices, partial = partial_query_block(block, x.clone(), mask, **kwargs)
    assert torch.equal(indices, torch.arange(16, device=device))
    torch.testing.assert_close(partial, dense, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_sparse_mask_returns_only_selected_tokens():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(9)
    device = torch.device("cuda")
    dim, heads = 48, 3
    block = WanAttentionBlock(
        "t2v_cross_attn", dim, 96, heads, qk_norm=True
    ).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 16, dim, device=device, dtype=torch.bfloat16)
    mask = torch.zeros(1, 16, 1, dtype=torch.bool, device=device)
    mask[:, [1, 5, 11]] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        indices, output = partial_query_block(
            block,
            x,
            mask,
            e=torch.randn(1, 6, dim, device=device, dtype=torch.float32),
            seq_lens=torch.tensor([16], dtype=torch.long),
            grid_sizes=torch.tensor([[1, 4, 4]], dtype=torch.long),
            freqs=rope_params(16, dim // heads).to(device),
            context=torch.randn(1, 8, dim, device=device, dtype=torch.bfloat16),
            context_lens=None,
        )
    assert indices.tolist() == [1, 5, 11]
    assert output.shape == (1, 3, dim)
