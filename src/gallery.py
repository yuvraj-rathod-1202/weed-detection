"""Gallery images for the website: raw photo, model output with laser shots, and ground truth.

Visual language (only greys + the two meaning colours):
    outlined in white   = weed (predicted, or ground truth)
    filled light white  = crop (predicted, or ground truth)
    crosshair green     = shot hit a weed       red = shot hit a crop
    crosshair grey      = shot on soil          hollow grey circle = shot vetoed by the crop gate
Only datasets whose licence allows sharing images go in the gallery (Sugar Beets 2016, CottonWeedDet12) -
never CropAndWeed.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.targeting.points import point_in_mask
from src.visualization import draw_target

HIT_RGB = (46, 125, 50)      # site palette: --hit
CROP_RGB = (198, 40, 40)     # site palette: --crop
SOIL_RGB = (158, 158, 158)   # site palette: --soil
WHITE = (255, 255, 255)

SUGARBEET_CROP_BGR = (0, 0, 255)       # Sugar Beets 2016 label colour of sugar beet; every other colour is a weed
SUGARBEET_UNKNOWN_BGR = (150, 150, 150)  # "unknown" plants: neither weed nor crop (left out, like ignore regions)

# Gallery photos, hand-picked to show the models working (the site says so). Sugar Beets 2016: among the 30
# frames with the most large weeds (>= 1500 px) and >= 2 sugar beets, those where the final model hits most
# weeds with at most 1 crop shot, checked by eye. CottonWeedDet12: test photos with >= 2 weeds that exp 2 hits
# fully, preferring ones exp 4 also does well on.
SUGARBEETS_FRAMES = (116, 136, 126, 102, 164)
COTTON_IMAGES = ("20210903_NIKOND3300_YL_120", "180_20210910_NIKOND3300_YL_219", "90_20210913_iPhoneSE_YL_84",
                 "90_20210827_iPhone11Pro_BT_94", "180_20210912_CanonEOS4000D_YL_12")
# Honesty check: 5 random Sugar Beets frames, random.Random(0).sample(sorted(all 283 frames), 5).
RANDOM_SUGARBEETS_FRAMES = (43, 155, 220, 238, 284)


def shot_outcome(point, gt_weed_masks: list[np.ndarray], gt_crop_masks: list[np.ndarray]) -> str:
    if any(point_in_mask(point, m) for m in gt_weed_masks):
        return "weed"
    if any(point_in_mask(point, m) for m in gt_crop_masks):
        return "crop"
    return "soil"


def sugarbeets_masks(label_bgr: np.ndarray, min_area_px: int = 50) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(weed masks, crop masks) from a Sugar Beets 2016 colour label: one mask per connected plant blob."""
    plant = label_bgr.any(axis=2)
    crop = (label_bgr == SUGARBEET_CROP_BGR).all(axis=2)
    weed = plant & ~crop & ~(label_bgr == SUGARBEET_UNKNOWN_BGR).all(axis=2)
    return _components(weed, min_area_px), _components(crop, min_area_px)


def render_output(image_rgb: np.ndarray, weed_masks: list[np.ndarray], crop_masks: list[np.ndarray],
                  shots: list[dict]) -> np.ndarray:
    """Predicted plants + laser shots; `shots` items: {"x", "y", "outcome": weed|crop|soil, "vetoed": bool}."""
    out = _draw_plants(image_rgb, weed_masks, crop_masks)
    thickness = _thickness(out)
    colours = {"weed": HIT_RGB, "crop": CROP_RGB, "soil": SOIL_RGB}
    for shot in shots:
        center = (int(round(shot["x"])), int(round(shot["y"])))
        if shot["vetoed"]:
            cv2.circle(out, center, thickness * 8, SOIL_RGB, thickness)
        else:
            draw_target(out, (shot["x"], shot["y"]), colours[shot["outcome"]], thickness)
    return out


def render_gt(image_rgb: np.ndarray, gt_weed_masks: list[np.ndarray], gt_crop_masks: list[np.ndarray]) -> np.ndarray:
    return _draw_plants(image_rgb, gt_weed_masks, gt_crop_masks)


def _draw_plants(image_rgb, weed_masks, crop_masks) -> np.ndarray:
    out = image_rgb.copy()
    thickness = _thickness(out)
    for mask in crop_masks:
        out[mask] = (0.6 * out[mask] + 0.4 * np.array(WHITE)).astype(np.uint8)
    for mask in weed_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, WHITE, thickness)
    return out


def _components(mask: np.ndarray, min_area_px: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]


def _thickness(image: np.ndarray) -> int:
    return max(2, round(max(image.shape[:2]) / 500))
