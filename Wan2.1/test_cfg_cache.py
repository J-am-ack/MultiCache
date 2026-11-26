"""Regression check using a small, randomly initialized Wan model and real block calls."""
from types import SimpleNamespace
import torch


def check(wan_module, apply):
    # Keep every sequence unpadded so native SDPA is usable on Windows.
    from importlib import import_module
    attention = import_module(wan_module.__package__ + ".attention")
    if not (attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE):
        wan_module.flash_attention = attention.attention
    torch.manual_seed(42)
    net = wan_module.WanModel(dim=96, num_heads=4, num_layers=2, ffn_dim=192, text_len=8,
                              text_dim=32, in_dim=4, out_dim=4, freq_dim=32).eval().cuda()
    torch.nn.init.normal_(net.head.head.weight, std=0.01)
    args = SimpleNamespace(task="t2v-1.3B", cfg_cache=True, cfg_tolerance=100, cfg_max_age=3,
        sample_guide_scale=6, sencache_thresh_start=0, sencache_thresh_main=0, sencache_K=1,
        spatial_cache=False, hybrid_cache=False, partial_query_attention=False,
        spatial_cache_block="1,4,4", spatial_cache_ratio=.25, hybrid_full_multiplier=2,
        hybrid_min_ratio=.1, sample_steps=8)
    apply(SimpleNamespace(model=net), args, 0)
    hits = []
    net.blocks[0].register_forward_hook(lambda *a: hits.append(1))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for sample in range(2):
            total = 0
            for step in range(8):
                # Force paired skips on steps 3 and 7, including the reset boundary.
                net.sencache_threshold_main = 1e15 if step in (3,7) else 0
                x = [torch.randn(4, 2, 4, 4, device="cuda")]
                t = torch.tensor([800.-step], device="cuda")
                calls = len(hits)
                pred = []
                for branch in range(2):
                    pred.append(net(x, t, [torch.randn(8, 32, device="cuda")], 8)[0])
                cost = len(hits) - calls
                assert torch.isfinite(pred[0]).all() and torch.isfinite(pred[1]).all()
                expected = [2,2,1,0,1,2,1,0][step]
                assert cost == expected, (step, cost, expected)
                total += cost
            assert net.cnt == 0 and net.cfg_cache.delta is None and total == 9
    print("PASS: two independent 8-step samples, observed block-stack calls 2,2,1,0,1,2,1,0.")


if __name__ == "__main__":
    from wan.modules import model
    from sencache import _apply_sencache
    check(model, _apply_sencache)

