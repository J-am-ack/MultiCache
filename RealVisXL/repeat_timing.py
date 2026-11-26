"""Repeat one prompt to show the run-to-run timing spread on the same initial noise."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from bench import CASES, sample, write_json


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
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    prompt = CASES["test"][0]["prompt"]
    sample(pipe, prompt, 31, "native20")
    rows = []
    modes = ["native20", "native14", "fixed", "B", "D"]
    for i in range(len(modes)):
        for mode in modes[i:] + modes[:i]:
            _, row, _ = sample(pipe, prompt, 42, mode, profile)
            rows.append(dict(repeat=i + 1, mode=mode, seconds=row["seconds"]))
            write_json(args.output, rows)
        print(f"Timing round {i+1} complete", flush=True)


if __name__ == "__main__":
    main()
