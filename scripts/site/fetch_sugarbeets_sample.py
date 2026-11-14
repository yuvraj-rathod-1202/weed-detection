#!/usr/bin/env python
"""Fetch a small random sample of Sugar Beets 2016 (Univ. Bonn, CC BY-SA 4.0) photos with their labels.

Downloads the 3.7 MB multiclass label archive, picks frames at random (seed 0), and reads only those RGB
photos out of the 1.6 GB raw-data zip with HTTP range requests. Label frame N belongs to photo
rgb_{N:05d}.png (checked: vegetation IoU 0.72-0.89 for the matching photo, < 0.15 for its neighbours).

    data/raw/sugarbeets2016_sample/labels/<frame>.png   colour labels (red = sugar beet, other colours = weeds)
    data/raw/sugarbeets2016_sample/rgb/<frame>.png      RGB photos (JAI camera, 1296 x 966)

Usage: python scripts/site/fetch_sugarbeets_sample.py [--count 5] [--seed 0]
Citation: Chebrolu et al., "Agricultural robot dataset for plant classification, localization and mapping on
sugar beet fields", IJRR 2017. https://www.ipb.uni-bonn.de/data/sugarbeets2016/
"""
from __future__ import annotations

import argparse
import io
import random
import re
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.remote_zip import HttpFile

LABELS_URL = "https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/multiclass/annotations.zip"
RAW_URL = "https://www.ipb.uni-bonn.de/datasets_IJRR2017/raw_data/160523/bonirob_2016-05-23-10-37-10_0.zip"
RECORDING = "bonirob_2016-05-23-10-37-10_0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/sugarbeets2016_sample")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(parents=True, exist_ok=True)

    labels = zipfile.ZipFile(io.BytesIO(requests.get(LABELS_URL, timeout=120).content))
    by_frame = {int(m.group(1)): name for name in labels.namelist()
                if (m := re.search(rf"{RECORDING}_frame(\d+)_GroundTruth_color\.png$", name))}
    frames = sorted(random.Random(args.seed).sample(sorted(by_frame), args.count))
    raw = zipfile.ZipFile(HttpFile(RAW_URL))
    for frame in frames:
        (out / "labels" / f"{frame:05d}.png").write_bytes(labels.read(by_frame[frame]))
        rgb = out / "rgb" / f"{frame:05d}.png"
        if not rgb.exists():
            rgb.write_bytes(raw.read(f"{RECORDING}/camera/jai/rgb/rgb_{frame:05d}.png"))
        print("frame", frame)
    print(f"{len(frames)} frames in {out}")


if __name__ == "__main__":
    main()
