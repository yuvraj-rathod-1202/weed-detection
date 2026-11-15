#!/usr/bin/env python
"""Fetch the Sugar Beets 2016 (Univ. Bonn, CC BY-SA 4.0) photos the gallery uses, with their labels.

Downloads the 3.7 MB multiclass label archive and reads only the needed RGB photos out of the 1.6 GB raw-data
zip with HTTP range requests: the hand-picked gallery frames (src/gallery.py SUGARBEETS_FRAMES) and the 5 random
frames used as an honesty check (RANDOM_SUGARBEETS_FRAMES; drawn once with
random.Random(0).sample(sorted(frames), 5), which this script re-checks). Label frame N belongs to photo
rgb_{N:05d}.png (checked: vegetation IoU 0.72-0.89 for the matching photo, < 0.15 for its neighbours).

    data/raw/sugarbeets2016_sample/labels/<frame>.png   colour labels (red = sugar beet, other colours = weeds)
    data/raw/sugarbeets2016_sample/rgb/<frame>.png      RGB photos (JAI camera, 1296 x 966)

Usage: python scripts/site/fetch_sugarbeets_sample.py
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
from src.gallery import RANDOM_SUGARBEETS_FRAMES, SUGARBEETS_FRAMES

LABELS_URL = "https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/multiclass/annotations.zip"
RAW_URL = "https://www.ipb.uni-bonn.de/datasets_IJRR2017/raw_data/160523/bonirob_2016-05-23-10-37-10_0.zip"
RECORDING = "bonirob_2016-05-23-10-37-10_0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/sugarbeets2016_sample")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(parents=True, exist_ok=True)

    labels = zipfile.ZipFile(io.BytesIO(requests.get(LABELS_URL, timeout=120).content))
    by_frame = {int(m.group(1)): name for name in labels.namelist()
                if (m := re.search(rf"{RECORDING}_frame(\d+)_GroundTruth_color\.png$", name))}
    assert sorted(random.Random(0).sample(sorted(by_frame), 5)) == sorted(RANDOM_SUGARBEETS_FRAMES)
    frames = sorted(set(SUGARBEETS_FRAMES) | set(RANDOM_SUGARBEETS_FRAMES))
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
