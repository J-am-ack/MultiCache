"""Score saved lossless images, publish per-image data, paired intervals and browsable grids."""

import argparse
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPImageProcessor,
    CLIPTokenizer,
    CLIPModel,
)

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf8"))["test"]
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


def score_subject(image, phrase, proc, model):
    inputs = proc(images=image, text=phrase + ".", return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0]
    ids = inputs.input_ids[0]
    exclude = torch.tensor(
        proc.tokenizer.all_special_ids + [proc.tokenizer.convert_tokens_to_ids(".")],
        device=model.device,
    )
    words = inputs.attention_mask[0].bool() & ~torch.isin(ids, exclude)
    return logits[:, : ids.numel()].sigmoid()[:, words].mean(1).amax().item()


def average(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def paired_interval(rows, baseline, mode):
    # Sampling prompts as clusters keeps the three seeds of each prompt together.
    def grouped(m):
        return np.array(
            [
                average([r for r in rows if r["mode"] == m and r["case"] == c["id"]], "seconds")
                for c in CASES
            ]
        )

    ref, other = grouped(baseline), grouped(mode)
    draws = np.random.default_rng(2026).integers(0, len(CASES), size=(5000, len(CASES)))
    ratios = ref[draws].mean(1) / other[draws].mean(1)
    return np.quantile(ratios, [0.025, 0.975]).tolist()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("detector", type=Path)
    ap.add_argument("report", type=Path)
    ap.add_argument(
        "--clip",
        type=Path,
        help="Local openai/clip-vit-base-patch32 directory; also accepts official ViT-B-32.pt",
    )
    args = ap.parse_args()
    report = args.report
    report.mkdir(exist_ok=True, parents=True)
    rows = json.loads((args.run / "runs.json").read_text())
    by_key = {(r["case"], r["seed"], r["mode"]): r for r in rows}
    expected = {(c["id"], s, m) for c in CASES for s in (42, 137, 2026) for m in MODES}
    if set(by_key) != expected or len(rows) != len(expected):
        raise ValueError("Incomplete or duplicated pairs; finish generation first.")
    proc = AutoProcessor.from_pretrained(args.detector, local_files_only=True)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(args.detector, local_files_only=True)
        .eval()
        .cuda()
    )
    if args.clip:
        if (args.clip / "ViT-B-32.pt").is_file():
            clip = torch.jit.load(str(args.clip / "ViT-B-32.pt"), map_location="cuda:0").eval()
            encode_text, encode_image = clip.encode_text, clip.encode_image
        else:
            clip = CLIPModel.from_pretrained(args.clip, local_files_only=True).eval().cuda()
            encode_text, encode_image = clip.get_text_features, clip.get_image_features
        clip_proc = CLIPImageProcessor.from_pretrained(args.clip, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(args.clip, local_files_only=True)
    for case in CASES:
        if args.clip:
            ids = tokenizer(
                case["prompt"],
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True,
            ).input_ids.cuda()
            caption = torch.nn.functional.normalize(encode_text(ids).float())
        folder = report / "grids"
        folder.mkdir(exist_ok=True)
        sheet = Image.new("RGB", (len(MODES) * 150, 3 * 176), "white")
        draw = ImageDraw.Draw(sheet)
        for j, seed in enumerate((42, 137, 2026)):

            def load(mode):
                return Image.open(
                    args.run / "images" / by_key[case["id"], seed, mode]["file"]
                ).convert("RGB")

            reference = np.array(load("native20"))
            for i, mode in enumerate(MODES):
                row = by_key[case["id"], seed, mode]
                image = load(mode)
                pixels = np.array(image)
                mse = np.square(pixels.astype(np.float32) - reference).mean()
                row["mse"] = float(mse)
                row["psnr_db"] = float(10 * np.log10(255**2 / mse)) if mse else None
                row["ssim"] = float(
                    structural_similarity(reference, pixels, channel_axis=2, data_range=255)
                )
                row["subject"] = score_subject(image, case["subject"], proc, model)
                if args.clip:
                    batch = clip_proc(images=image, return_tensors="pt").pixel_values.cuda()
                    visual = torch.nn.functional.normalize(encode_image(batch).float())
                    row["clip_cosine"] = (visual * caption).sum().item()
                sheet.paste(
                    image.resize((150, 150), Image.Resampling.LANCZOS), (150 * i, 176 * j + 25)
                )
                draw.text((150 * i + 4, 176 * j + 4), f"{mode} / {seed}", fill="black")
        sheet.save(folder / (case["id"] + ".jpg"), quality=90)
        print("Scored", case["id"], flush=True)
    results = []
    for mode in MODES:
        group = [r for r in rows if r["mode"] == mode]
        out = {
            k: average(group, k)
            for k in ("seconds", "branches", "peak_mib", "ssim", "psnr_db", "subject")
        }
        out["mode"] = mode
        if args.clip:
            out["clip_cosine"] = average(group, "clip_cosine")
        out["speedup"] = (
            average([r for r in rows if r["mode"] == "native20"], "seconds") / out["seconds"]
        )
        out["speedup_ci"] = paired_interval(rows, "native20", mode)
        out["perfect_pairs"] = sum(r["mse"] == 0 for r in group)
        results.append(out)
    (report / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    (report / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    shutil.copyfile(args.run / "calibration.json", report / "calibration.json")
    samples = report / "samples"
    samples.mkdir(exist_ok=True)
    for case in CASES:
        with zipfile.ZipFile(samples / (case["id"] + ".zip"), "w") as archive:
            for row in (r for r in rows if r["case"] == case["id"]):
                with Image.open(args.run / "images" / row["file"]) as image:
                    data = io.BytesIO()
                    image.save(data, format="JPEG", quality=90)
                    archive.writestr(Path(row["file"]).with_suffix(".jpg").name, data.getvalue())


if __name__ == "__main__":
    main()
