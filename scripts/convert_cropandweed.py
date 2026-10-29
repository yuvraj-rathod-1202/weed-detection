#!/usr/bin/env python
"""Download (optional) and convert CropAndWeed into the raw layout prepare_data.py reads.

The dataset licence forbids redistribution, so each machine downloads it from the official
source. Archives are streamed straight into tar (no .tar file is kept on disk). Both steps are
resumable: re-run after an interruption.

Usage:
    # download annotations + image part 1 of 4 (~3 GB) and convert
    python scripts/convert_cropandweed.py --download --parts 1 --src data/raw/cropandweed_src --out data/raw/cropandweed
    # full dataset (~11.5 GB), e.g. on Kaggle
    python scripts/convert_cropandweed.py --download --src /tmp/cnw_src --out /tmp/cropandweed
    # already downloaded: convert only
    python scripts/convert_cropandweed.py --src data/raw/cropandweed_src --out data/raw/cropandweed
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from tqdm import tqdm

from src.data.cropandweed import convert_dataset

BASE_URL = "https://vitro-testing.com/wp-content/uploads/2022/12/cropandweed_{}.tar"


def download(src: Path, parts: list[int]) -> None:
    src.mkdir(parents=True, exist_ok=True)
    names = ["annotations"] + [f"images{p}of4" for p in parts]
    for name in names:
        marker = src / f".done_{name}"
        if marker.exists():
            print(f"{name}: already downloaded")
            continue
        print(f"{name}: downloading and extracting ...")
        with requests.get(BASE_URL.format(name), stream=True, timeout=60) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode="r|") as tar:
                safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}  # Python >= 3.10.12
                for member in tar:  # streamed: one pass, no seeking
                    tar.extract(member, src, **safe)
        marker.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Folder with (or for) the extracted official archives")
    parser.add_argument("--out", required=True, help="Output raw dataset folder (images/, labels/, classes.txt, ...)")
    parser.add_argument("--download", action="store_true", help="Download the official archives into --src first")
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4],
                        help="Image archives to download (default: all four)")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if args.download:
        download(src, args.parts)
    result = convert_dataset(src, out, progress=lambda items: tqdm(items, desc="Converting", unit="img"))
    print(f"{result['images']} images in {out} ({result['converted_now']} converted now)")


if __name__ == "__main__":
    main()
