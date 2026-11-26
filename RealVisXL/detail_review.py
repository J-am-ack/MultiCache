"""Build masked image sheets for a literal prompt check and collect the reviewer labels."""

import argparse
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw

RULES = {
    "cups": "Exactly two blue mugs and a visibly white handle on the right mug.",
    "sign": "The blue sign visibly spells PARK in white; other words are allowed.",
    "cubes": "Exactly three cubes in a row, ordered red, green, blue from left to right.",
}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("sheets", type=Path)
    ap.add_argument("--labels", type=Path, help="JSON object mapping anonymous image IDs to 0 or 1")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    args.sheets.mkdir(exist_ok=True, parents=True)
    rows = json.loads((args.run / "runs.json").read_text())
    mapping = {}
    for case in RULES:
        items = sorted((r for r in rows if r["case"] == case), key=lambda r: r["file"])
        random.Random("detail-" + case).shuffle(items)
        for start in range(0, len(items), 9):
            sheet = Image.new("RGB", (3 * 280, 3 * 304), "white")
            draw = ImageDraw.Draw(sheet)
            for k, row in enumerate(items[start : start + 9]):
                idx = start + k + 1
                key = f"{case}-{idx:02d}"
                mapping[key] = row["file"]
                x, y = k % 3 * 280, k // 3 * 304
                im = Image.open(args.run / "images" / row["file"])
                sheet.paste(im.resize((280, 280), Image.Resampling.LANCZOS), (x, y + 23))
                draw.text((x + 4, y + 4), key, fill="black")
            sheet.save(args.sheets / f"{case}-{start//9+1}.jpg", quality=93)
    if args.labels:
        if not args.report:
            ap.error("--labels requires --report")
        labels = json.loads(args.labels.read_text())
        if set(labels) != set(mapping) or any(value not in (0, 1) for value in labels.values()):
            raise ValueError("Need one binary label for every anonymous image")
        result = dict(rules=RULES, ratings={name: labels[k] for k, name in mapping.items()})
        args.report.write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
