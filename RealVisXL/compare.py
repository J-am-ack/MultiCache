"""Summarize paired runs, timing repeats, step rankings and masked detail scores."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import numpy as np


def compare_pairs(index, pairs, method, baseline):
    left = [index[case, seed, method] for case, seed in pairs]
    right = [index[case, seed, baseline] for case, seed in pairs]
    deltas = [a["ssim"] - b["ssim"] for a, b in zip(left, right)]

    def refreshes(run):
        return [i for i, action in enumerate(run["trace"]) if action in ("full", "single")]

    cases = sorted({case for case, _ in pairs})
    def by_case(group):
        return np.array([mean(r["seconds"] for r in group if r["case"] == case) for case in cases])

    draws = np.random.default_rng(2026).integers(0, len(cases), size=(5000, len(cases)))
    ratios = by_case(right)[draws].mean(1) / by_case(left)[draws].mean(1)
    return {
        "ssim_delta": mean(deltas),
        "wins": sum(d > 1e-8 for d in deltas),
        "ties": sum(abs(d) <= 1e-8 for d in deltas),
        "speed_ratio": mean(r["seconds"] for r in right) / mean(r["seconds"] for r in left),
        "speed_ratio_ci": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "same_schedule": sum(refreshes(a) == refreshes(b) for a, b in zip(left, right)),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("report", type=Path)
    folder = parser.parse_args().report

    def read(name):
        return json.loads((folder / name).read_text(encoding="utf8"))

    runs = read("runs.json")
    modes = [r["mode"] for r in read("summary.json")]
    pairs = sorted({(r["case"], r["seed"]) for r in runs})
    index = {(r["case"], r["seed"], r["mode"]): r for r in runs}
    calibration = read("calibration.json")["observations"]
    audit = read("step_audit.json")
    review = read("detail.json")
    repeats = read("repeat.json")
    choices = [("A", "global"), ("B", "global"), ("D", "fixed"), ("C", "fixed"),
               ("C_guard", "C"), ("A", "native14"), ("B", "native14")]
    out = {
        "compare": {f"{m}-{ref}": compare_pairs(index, pairs, m, ref) for m, ref in choices},
        "actions": {m: dict(Counter(a for r in runs if r["mode"] == m for a in r["trace"])) for m in modes},
        "calibrate_s": sum(r["seconds"] for r in calibration),
        "early_mse": float(np.mean([r["mse"][1:7] for r in calibration])),
        "late_mse": float(np.mean([r["mse"][13:19] for r in calibration])),
        "audit": {
            k: {field: mean(c["ranking"][k][field] for c in audit.values())
                for field in ("spearman", "worst6_found")}
            for k in ("global_score", "local_score", "weighted_score")
        },
        "detail": {
            m: sum(review["ratings"][r["file"]] for r in runs
                   if r["mode"] == m and r["case"] in review["rules"])
            for m in modes
        },
        "repeat": {
            m: np.quantile([r["seconds"] for r in repeats if r["mode"] == m], [0, .5, 1]).tolist()
            for m in sorted({r["mode"] for r in repeats})
        },
        "group_quality": {
            m: {case: mean(r["ssim"] for r in runs if r["case"] == case and r["mode"] == m)
                for case in sorted({r["case"] for r in runs})}
            for m in modes
        },
    }
    (folder / "comparisons.json").write_text(json.dumps(out, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
