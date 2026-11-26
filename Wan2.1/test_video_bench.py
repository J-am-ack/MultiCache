"""Integration smoke test for bench_video.sample using the official, reduced-size WanModel."""

from types import SimpleNamespace
import torch
from wan.modules import model as wan_model, attention
from bench_video import FIXED, sample, _parse_args


@torch.inference_mode()
def main():
    if not (attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE):
        wan_model.flash_attention = attention.attention
    torch.manual_seed(37)
    net = (
        wan_model.WanModel(
            dim=96,
            num_heads=4,
            num_layers=2,
            ffn_dim=192,
            text_len=8,
            text_dim=32,
            in_dim=4,
            out_dim=4,
            freq_dim=32,
        )
        .eval()
        .cuda()
    )
    torch.nn.init.normal_(net.head.head.weight, std=0.01)
    calls = []
    net.blocks[0].register_forward_hook(lambda *args: calls.append(1))

    def generate(prompt, sampling_steps, seed, **kwargs):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn(4, 2, 4, 4, device="cuda", generator=generator)
        contexts = [torch.randn(8, 32, device="cuda", generator=generator) for _ in range(2)]
        for t in torch.linspace(999, 0, sampling_steps, device="cuda"):
            c, u = [net([latent], t.reshape(1), [ctx], 8)[0] for ctx in contexts]
            latent = latent - 0.01 * (u + 6 * (c - u))
        return latent

    pipe = SimpleNamespace(model=net, generate=generate)
    args = _parse_args(
        [
            "--ckpt_dir",
            ".",
            "--task",
            "t2v-1.3B",
            "--size",
            "832*480",
            "--sample_steps",
            "20",
            "--sample_guide_scale",
            "6",
            "--cfg_tolerance",
            "100",
        ]
    )
    profile = dict(
        importance=[1.0] * 20, cutoffs={s: [1.0] * 6 for s in ["global", "local", "weighted"]}
    )
    orig = type(net).forward
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for mode in [
            "native20",
            "sencache",
            "calibrate",
            "probe",
            "fixed",
            "global",
            "A",
            "B",
            "C",
            "C_guard",
            "D",
            "sen_C",
            "sen_D",
        ]:
            calls.clear()
            video, stats, scores = sample(
                pipe, args, "fixture", 42, mode, profile, probe=4, replay=FIXED
            )
            assert torch.isfinite(video).all() and stats["branches"] == len(calls), mode
            assert type(net).forward is orig and net.bench_policy is None
            if mode in ("global", "A", "B", "fixed", "C", "C_guard", "sen_C"):
                assert len(calls) == 28, mode
            if mode in ("D", "sen_D"):
                assert 14 < len(calls) < 28, mode
            if mode == "calibrate":
                assert len(scores) == 20 and len(calls) == 40
            if mode == "probe":
                assert len(calls) == 38
            print(f"{mode}: {len(calls)} verified block-stack calls")


if __name__ == "__main__":
    main()
