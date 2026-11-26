"""Run the paired RealVisXL benchmark with a local diffusers checkpoint."""

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "Wan2.1"))
from cfg_cache import CFGCache
from budget_policy import WindowBudget, fit_profiles

STEPS = 20
SCALE = 6.0
SEEDS = [42, 137, 2026]
MODES = [
    "native20",
    "native16",
    "native14",
    "serial20",
    "fixed",
    "global",
    "A",
    "B",
    "C",
    "C_guard",
    "D",
]
CASES = json.loads((ROOT / "cases.json").read_text(encoding="utf8"))


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    temp.replace(path)


def drift(x, anchor):
    error = (x.float() - anchor.float()).square().mean(dim=1, keepdim=True)
    blocks = F.avg_pool2d(error, 8, stride=8).flatten()
    global_rms = blocks.mean().sqrt()
    local_rms = blocks.topk(max(1, len(blocks) // 8)).values.mean().sqrt()
    return global_rms.item(), ((global_rms + local_rms) * 0.5).item()


class Policy:
    def __init__(self, mode, calibration=None, probe_step=None):
        self.mode = mode
        self.calibration = calibration
        self.probe_step = probe_step
        self.trace = []
        self.scores = []
        self.last = self.previous = self.anchor = None
        self.last_i = self.previous_i = -1
        self.cfg = CFGCache(SCALE, 0.35, 3)
        self.budget = WindowBudget()

    def skip(self, i, x):
        if self.mode == "probe":
            return i == self.probe_step
        if self.mode in ("fixed", "C", "C_guard", "D"):
            return i in (2, 5, 8, 11, 14, 17)
        if self.mode not in ("global", "A", "B") or not self.budget.eligible(i):
            return False
        global_score, local_score = drift(x, self.anchor)
        return self.budget.choose(i, self.mode, global_score, local_score, self.calibration)

    def reuse(self, i):
        if self.mode not in ("C", "C_guard") or self.previous is None:
            return self.last, "reuse"
        ratio = (i - self.last_i) / (self.last_i - self.previous_i)
        update = ratio * (self.last.float() - self.previous.float())
        if self.mode == "C_guard" and (
            self.last_i - self.previous_i > 2 or update.norm() > 0.15 * self.last.float().norm()
        ):
            return self.last, "fallback"
        return (self.last + update).to(self.last.dtype), "predict"

    @contextmanager
    def attach(self, pipe):
        original = pipe.unet.forward

        def forward(x, t, encoder_hidden_states, added_cond_kwargs, **kwargs):
            i = len(self.trace)
            if self.mode == "calibrate":
                self.scores.append(
                    drift(x[:1], self.anchor) if self.anchor is not None else (0.0, 0.0)
                )
            if self.skip(i, x[:1]):
                out, action = self.reuse(i)
            else:
                action = self.cfg.choose(i, False, STEPS - 1) if self.mode == "D" else "full"

                def branch(j):
                    added = {key: v[j : j + 1] for key, v in added_cond_kwargs.items()}
                    return original(
                        x[j : j + 1],
                        t,
                        encoder_hidden_states=encoder_hidden_states[j : j + 1],
                        added_cond_kwargs=added,
                        **kwargs,
                    )[0]

                if action == "single":
                    c = branch(1)
                    out = torch.cat([self.cfg.estimate_unconditional(c.float()).to(c.dtype), c])
                elif self.mode == "serial20":
                    out = torch.cat([branch(0), branch(1)])
                else:
                    out = original(
                        x,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        **kwargs,
                    )[0]
                if self.mode == "D" and action == "full":
                    u, c = out.float().chunk(2)
                    self.cfg.observe_pair(i, c, u)
                self.previous, self.previous_i = self.last, self.last_i
                self.last, self.last_i = out, i
                self.anchor = x[:1]
            self.trace.append(action)
            return (out,)

        pipe.unet.forward = forward
        try:
            yield
        finally:
            pipe.unet.forward = original


def sample(pipe, prompt, seed, mode, calibration=None, probe_step=None):
    policy = Policy(mode, calibration, probe_step)
    native = mode.startswith("native")
    steps = int(mode[6:]) if native else STEPS
    with nullcontext() if native else policy.attach(pipe):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        image = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=steps,
            guidance_scale=SCALE,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20
    trace = ["full"] * steps if native else policy.trace
    return (
        image,
        dict(
            seconds=elapsed,
            peak_mib=peak,
            trace=trace,
            branches=2 * trace.count("full") + trace.count("single"),
        ),
        policy.scores,
    )


def mse(image, reference):
    return float(
        np.square(
            np.asarray(image, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
        ).mean()
    )


def calibrate(pipe, folder):
    path = folder / "calibration.json"
    if path.exists():
        return json.loads(path.read_text())
    observations = []
    for case in CASES["calibration"]:
        checkpoint = folder / ("cal_" + case["id"] + ".json")
        records = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        ref, info, scores = sample(pipe, case["prompt"], 31, "calibrate")
        for i in range(1, 19):
            if str(i) in records:
                continue
            candidate, timing, _ = sample(pipe, case["prompt"], 31, "probe", probe_step=i)
            records[str(i)] = dict(mse=mse(candidate, ref), seconds=timing["seconds"])
            write_json(checkpoint, records)
        observations.append(
            dict(
                scores=scores,
                mse=[0] + [records[str(i)]["mse"] for i in range(1, 19)] + [0],
                seconds=info["seconds"] + sum(r["seconds"] for r in records.values()),
            )
        )
    data = fit_profiles(observations)
    write_json(path, data)
    return data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    folder = args.output
    folder.mkdir(exist_ok=True, parents=True)
    images = folder / "images"
    images.mkdir(exist_ok=True)
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
    sample(pipe, CASES["calibration"][0]["prompt"], 0, "native20")
    calibration = calibrate(pipe, folder)
    path = folder / "runs.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [row for row in rows if (images / row["file"]).is_file()]
    done = {r["file"] for r in rows}
    for case in CASES["test"]:
        for seed in SEEDS:
            modes = MODES.copy()
            random.Random(case["id"] + str(seed)).shuffle(modes)
            for mode in modes:
                name = f"{case['id']}_{seed}_{mode}.png"
                if name in done and (images / name).is_file():
                    continue
                image, row, _ = sample(pipe, case["prompt"], seed, mode, calibration)
                image.save(images / name)
                rows.append(dict(case=case["id"], seed=seed, mode=mode, file=name, **row))
                write_json(path, rows)
                done.add(name)
                print(f"{len(rows)} {name} {row['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
