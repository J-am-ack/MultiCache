"""Draw the latency/fidelity trade-off and calibration sensitivity from measured JSON."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("report", type=Path)
    out = parser.parse_args().report
    rows = json.loads((out / "summary.json").read_text())
    colors = {
        "native": "#66717E",
        "serial": "#9DA4A8",
        "fixed": "#A26D42",
        "global": "#506B9C",
        "A": "#277867",
        "B": "#69933A",
        "C": "#A65C8F",
        "D": "#BC673D",
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    for row in rows:
        name = row["mode"]
        if name == "serial20":
            continue
        key = "native" if name.startswith("native") else name.split("_")[0]
        lo, hi = row["speedup_ci"]
        x = row["speedup"]
        ax.errorbar(
            x,
            row["ssim"],
            xerr=[[max(x - lo, 0)], [max(hi - x, 0)]],
            fmt="D" if name == "C_guard" else "o",
            color=colors[key],
            capsize=3,
            markerfacecolor="white" if name == "C_guard" else colors[key],
            label=name,
        )
    ax.set(
        xlabel="End-to-end speedup relative to native 20 steps (higher is faster)",
        ylabel="SSIM against the paired native 20-step image",
        title="RealVisXL V4.0   |   36 prompt-seed pairs",
    )
    ax.legend(ncol=2, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.16)
    fig.savefig(out / "speed-quality.png", dpi=170)
    plt.close(fig)
    cal = json.loads((out / "calibration.json").read_text())
    fig, ax = plt.subplots(figsize=(8.3, 3.6), constrained_layout=True)
    for row, label, color in zip(
        cal["observations"], ["Bicycle", "Violin"], ["#277867", "#B77543"]
    ):
        ax.plot(range(1, 19), row["mse"][1:19], label=label, color=color, lw=2, marker=".", ms=7)
    ax.set(
        xlabel="Single skipped step (zero-based)",
        ylabel="Final-image pixel MSE",
        title="How far does one skipped step move the final image?",
    )
    ax.set_xticks(range(1, 19))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "skip-impact.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
