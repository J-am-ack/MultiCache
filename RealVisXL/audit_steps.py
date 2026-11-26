"""Measure how well a cheap skip score ranks actual guided-output errors along a dense trajectory."""

import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import torch
from diffusers import StableDiffusionXLPipeline
from bench import CASES, SCALE, STEPS, drift, write_json
import json


@torch.inference_mode()
def audit(pipe, prompt, profile):
    original = pipe.unet.forward
    rows = []
    last_x = last_g = None

    def observe(x, t, **kwargs):
        nonlocal last_x, last_g
        out = original(x, t, **kwargs)
        u, c = out[0].float().chunk(2)
        guided = u + SCALE * (c - u)
        step = len(rows)
        row = dict(step=step)
        if last_x is not None:
            g, local = drift(x[:1], last_x)
            row.update(
                global_score=g,
                local_score=local,
                weighted_score=g * profile["importance"][step],
                error=((guided - last_g).norm() / guided.norm().clamp_min(1e-8)).item(),
            )
        rows.append(row)
        last_x, last_g = x[:1], guided
        return out

    pipe.unet.forward = observe
    try:
        pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=STEPS,
            guidance_scale=SCALE,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="latent",
        )
    finally:
        pipe.unet.forward = original
    candidates = rows[1:19]
    errors = np.array([r["error"] for r in candidates])
    worst = set(np.argsort(errors)[-6:])
    stats = {}
    for name in ("global_score", "local_score", "weighted_score"):
        values = np.array([r[name] for r in candidates])
        stats[name] = {
            "spearman": float(spearmanr(values, errors).statistic),
            "worst6_found": len(worst & set(np.argsort(values)[-6:])),
        }
    return dict(steps=rows, ranking=stats)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("model")
    ap.add_argument("profile", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    profile = json.loads(args.profile.read_text())
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    results = {}
    for case in CASES["test"]:
        results[case["id"]] = audit(pipe, case["prompt"], profile)
        write_json(args.output, results)
        print(case["id"], flush=True)


if __name__ == "__main__":
    main()
