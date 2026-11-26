"""Paired 17-frame Wan benchmark. Run from the team's existing Wan Python environment."""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from budget_policy import fit_profiles
from video_policy import VideoPolicy
from sencache import _apply_sencache, _parse_args, _create_pipeline, WAN_CONFIGS, cache_video

HERE = Path(__file__).resolve().parent
FIXED = ["reuse" if i in (2, 5, 8, 11, 14, 17) else "full" for i in range(20)]
MODES = ["native14", "fixed", "global", "A", "B", "C", "C_guard", "D", "sen_C", "sen_D"]
CAL = ["A red paper kite flying above a green field.", "A ferry crossing a river at sunset."]


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    temp.replace(path)


@contextmanager
def configured(pipe, args, mode, profile, probe, replay):
    model = pipe.model
    cls = type(model)
    original = cls.forward
    control = None
    try:
        if not mode.startswith("native"):
            args.cfg_cache = mode in ("D", "sen_D")
            _apply_sencache(pipe, args, 0)
            internal = {"sen_D": "D", "sen_C": "C_guard"}.get(mode, mode)
            plan = (
                replay
                if mode.startswith("sen_")
                else FIXED if mode in ("C", "C_guard", "D") else None
            )
            control = VideoPolicy(internal, profile, probe, plan)
            model.bench_policy = control
        yield control
    finally:
        cls.forward = original
        model.bench_policy = None
        model.cfg_cache = None


def sample(pipe, args, prompt, seed, mode, profile=None, probe=None, replay=None):
    steps = int(mode[6:]) if mode.startswith("native") else 20
    with configured(pipe, args, mode, profile, probe, replay) as control:
        torch.cuda.synchronize()
        start = time.perf_counter()
        video = pipe.generate(
            prompt,
            size=(832, 480),
            frame_num=args.frame_num,
            shift=8.0,
            sample_solver=args.sample_solver,
            sampling_steps=steps,
            guide_scale=6.0,
            seed=seed,
            offload_model=True,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        trace = control.trace if control is not None else ["full"] * steps
        if len(trace) != steps:
            raise RuntimeError(f"Expected {steps} conditional passes, observed {len(trace)}")
        return (
            video.detach().float().cpu(),
            dict(
                seconds=seconds,
                trace=trace,
                branches=2 * trace.count("full") + trace.count("single"),
            ),
            control.scores if control else [],
        )


def calibrate(pipe, args, out):
    path = out / "video_profile.json"
    if path.exists():
        return json.loads(path.read_text())
    observations = []
    for j, prompt in enumerate(CAL):
        ref, info, scores = sample(pipe, args, prompt, 31, "calibrate")
        damage = [0.0]
        seconds = info["seconds"]
        for step in range(1, 19):
            video, info, _ = sample(pipe, args, prompt, 31, "probe", probe=step)
            damage.append((video - ref).square().mean().item() * 127.5**2)
            seconds += info["seconds"]
            print(f"Calibration prompt {j+1}, step {step}/18", flush=True)
        observations.append(dict(scores=scores, mse=damage + [0.0], seconds=seconds))
    profile = fit_profiles(observations)
    save(path, profile)
    return profile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=HERE / "test_prompt.json",
        help="One plain-text prompt or JSON object with prompt per line",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Compare dense, original SenCache and D on its refresh steps",
    )
    opt = parser.parse_args()
    opt.checkpoint = opt.checkpoint.resolve()
    out = opt.output.resolve()
    prompts = []
    for line in opt.prompts.read_text(encoding="utf8").splitlines():
        if line.strip():
            prompts.append(
                json.loads(line)["prompt"] if line.lstrip().startswith("{") else line.strip()
            )
    if not prompts or any(p in CAL for p in prompts):
        parser.error("Provide test prompts distinct from the calibration set")
    out.mkdir(exist_ok=True, parents=True)
    os.chdir(HERE)
    args = _parse_args(
        [
            "--ckpt_dir",
            str(opt.checkpoint),
            "--task",
            "t2v-1.3B",
            "--size",
            "832*480",
            "--t5_cpu",
            "--sample_steps",
            "20",
            "--frame_num",
            "17",
            "--sample_guide_scale",
            "6",
            "--offload_model",
            "True",
            "--cfg_tolerance",
            "0.35",
        ]
    )
    config = WAN_CONFIGS[args.task]
    pipe = _create_pipeline(args, config, 0)
    sample(pipe, args, CAL[0], 0, "native14")
    profile = None if opt.quick else calibrate(pipe, args, out)
    rows = []
    for j, prompt in enumerate(prompts):
        for seed in opt.seeds:
            order = ["sen_D"] if opt.quick else MODES.copy()
            random.Random(f"{j}-{seed}").shuffle(order)
            replay = None
            for mode in ["native20", "sencache", *order]:
                video, row, _ = sample(pipe, args, prompt, seed, mode, profile, replay=replay)
                if mode == "native20":
                    ref = video
                if mode == "sencache":
                    replay = row["trace"]
                mse = (ref - video).square().mean().item()
                frame_error = video - ref
                temporal_rmse = frame_error.diff(dim=1).square().mean().sqrt().item() * 127.5
                filename = f"{j:03d}_{seed}_{mode}.mp4"
                result = cache_video(
                    video[None],
                    save_file=str(out / filename),
                    fps=config.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if result is None or not (out / filename).is_file():
                    raise RuntimeError(f"Video encoder failed for {filename}: {result}")
                rows.append(
                    dict(
                        prompt=prompt,
                        seed=seed,
                        mode=mode,
                        file=filename,
                        **row,
                        psnr_db=10 * math.log10(4 / mse) if mse else None,
                        temporal_rmse_255=temporal_rmse,
                    )
                )
                save(out / "video_runs.json", rows)
                print(
                    f"{filename}: {row['seconds']:.2f}s, {row['branches']} dense branches",
                    flush=True,
                )


if __name__ == "__main__":
    main()
