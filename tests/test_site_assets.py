"""The public site may only contain images whose licence allows sharing (never CropAndWeed photos)."""
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
ALLOWED_DATASETS = {"Sugar Beets 2016", "CottonWeedDet12"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
CROPANDWEED_NAME = re.compile(r"(ave|vwg)-\d{4}-\d{4}")


def _site_images():
    for path in SITE.rglob("*"):
        rel = path.relative_to(SITE)
        if rel.parts and rel.parts[0] in {"_site", ".quarto"}:
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield rel.as_posix()


def _gallery():
    return json.loads((SITE / "data" / "gallery.json").read_text(encoding="utf-8"))


def test_gallery_has_an_output_for_every_model_on_every_photo():
    gallery = _gallery()
    keys = [m["key"] for m in gallery["models"]]
    assert keys == ["baseline", "exp2", "exp3", "exp4", "exp5"] and gallery["default"] == "exp4"
    for item in gallery["items"]:
        assert sorted(item["outputs"]) == sorted(keys), item["id"]
        for out in item["outputs"].values():
            assert {"file", "weeds_hit", "crop_shots", "soil_shots", "vetoed"} <= set(out)
            assert 0 <= out["weeds_hit"] <= item["weeds"]


def test_site_images_are_only_permitted_gallery_files():
    gallery = _gallery()
    permitted = {}
    for item in gallery["items"]:
        files = [item["raw"], item["gt"]] + [out["file"] for out in item["outputs"].values()]
        permitted.update({f: item["dataset"] for f in files})
    for image in _site_images():
        assert not CROPANDWEED_NAME.search(image), f"CropAndWeed image in site/: {image}"
        assert image in permitted, f"{image} is not a gallery file listed in site/data/gallery.json"
        assert permitted[image] in ALLOWED_DATASETS, f"{image} comes from {permitted[image]}"
