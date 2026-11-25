"""Partial-query execution for Wan transformer blocks.

Selected query tokens attend to keys and values from the full token sequence.
This is an experimental sparse-refresh backend, not part of AdaCache.
"""

import torch
import torch.cuda.amp as amp

from wan.modules.attention import flash_attention
from wan.modules.model import rope_apply


def _rope_apply_selected(q, grid_sizes, freqs, token_indices):
    """Apply Wan's 3-D RoPE at arbitrary flattened token positions (batch=1)."""
    if q.size(0) != 1 or grid_sizes.size(0) != 1:
        raise ValueError("partial-query currently supports batch size 1")
    frames, height, width = grid_sizes[0].tolist()
    head_dim_half = q.size(3) // 2
    freq_parts = freqs.split([
        head_dim_half - 2 * (head_dim_half // 3),
        head_dim_half // 3,
        head_dim_half // 3,
    ], dim=1)
    position_freqs = torch.cat([
        freq_parts[0][:frames].view(frames, 1, 1, -1).expand(
            frames, height, width, -1),
        freq_parts[1][:height].view(1, height, 1, -1).expand(
            frames, height, width, -1),
        freq_parts[2][:width].view(1, 1, width, -1).expand(
            frames, height, width, -1),
    ], dim=-1).reshape(frames * height * width, 1, -1)
    selected_freqs = position_freqs.index_select(0, token_indices)
    q_complex = torch.view_as_complex(
        q[0].to(torch.float64).reshape(q.size(1), q.size(2), -1, 2))
    return torch.view_as_real(q_complex * selected_freqs).flatten(2)[None].float()


def partial_query_block(
    block,
    x,
    refresh_token_mask,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
):
    """Evaluate one Wan block only for selected query/output tokens.

    K/V projections remain dense so selected queries retain global self-attention
    context. Returns ``(selected_indices, selected_block_output)``.
    """
    if x.size(0) != 1:
        raise ValueError("partial-query currently supports batch size 1")
    indices = refresh_token_mask[0, :, 0].nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return indices, x[:, :0]

    with amp.autocast(dtype=torch.float32):
        modulation = (block.modulation + e).chunk(6, dim=1)

    normalized = block.norm1(x).float() * (1 + modulation[1]) + modulation[0]
    selected_normalized = normalized.index_select(1, indices)
    batch, selected_len = x.size(0), indices.numel()
    heads, head_dim = block.num_heads, block.dim // block.num_heads

    attention = block.self_attn
    q = attention.norm_q(attention.q(selected_normalized)).view(
        batch, selected_len, heads, head_dim)
    k = attention.norm_k(attention.k(normalized)).view(
        batch, x.size(1), heads, head_dim)
    v = attention.v(normalized).view(batch, x.size(1), heads, head_dim)
    q = _rope_apply_selected(q, grid_sizes, freqs, indices)
    k = rope_apply(k, grid_sizes, freqs)
    selected = flash_attention(
        q=q,
        k=k,
        v=v,
        k_lens=seq_lens,
        window_size=attention.window_size,
    )
    selected = attention.o(selected.flatten(2))

    selected_x = x.index_select(1, indices)
    with amp.autocast(dtype=torch.float32):
        selected_x = selected_x + selected * modulation[2]
    selected_x = selected_x + block.cross_attn(
        block.norm3(selected_x), context, context_lens)
    ffn_input = block.norm2(selected_x).float() * (
        1 + modulation[4]) + modulation[3]
    selected_ffn = block.ffn(ffn_input)
    with amp.autocast(dtype=torch.float32):
        selected_x = selected_x + selected_ffn * modulation[5]
    return indices, selected_x


def partial_query_stack(
    blocks,
    x,
    refresh_token_mask,
    block_residuals,
    **kwargs,
):
    """Run a block stack with dense cached residuals and sparse fresh queries."""
    if len(block_residuals) != len(blocks):
        raise ValueError("one cached residual is required for every transformer block")
    updated_residuals = []
    for block, cached_residual in zip(blocks, block_residuals):
        block_input = x
        indices, selected_output = partial_query_block(
            block, x, refresh_token_mask, **kwargs)
        x = x + cached_residual
        x[:, indices] = selected_output
        cached_residual[:, indices] = selected_output - block_input[:, indices]
        updated_residuals.append(cached_residual)
    return x, updated_residuals


def dense_stack_with_residuals(blocks, x, **kwargs):
    """Dense reference execution that also captures per-block residuals."""
    residuals = []
    for block in blocks:
        block_input = x
        x = block(x, **kwargs)
        residuals.append((x - block_input).detach())
    return x, residuals
