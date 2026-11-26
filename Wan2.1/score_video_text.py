"""After generation, compare the prompt with evenly spaced video frames using CLIP."""

import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("runs", type=Path, help="video_runs.json from bench_video.py")
    ap.add_argument("clip", type=Path, help="local Hugging Face openai/clip-vit-base-patch32")
    args = ap.parse_args()
    proc = CLIPProcessor.from_pretrained(args.clip, local_files_only=True)
    model = CLIPModel.from_pretrained(args.clip, local_files_only=True).eval().cuda()
    rows = json.loads(args.runs.read_text(encoding="utf8"))
    for row in rows:
        with imageio.get_reader(str(args.runs.parent / row["file"]), "ffmpeg") as reader:
            length = reader.count_frames()
            frames = [reader.get_data(int(i)) for i in np.linspace(0, length - 1, min(5, length))]
        inputs = proc(
            text=[row["prompt"]], images=frames, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        out = model(**inputs)
        row["frame_clip_cosine"] = (out.image_embeds @ out.text_embeds.T).mean().item()
    path = args.runs.with_name("video_scores.json")
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
