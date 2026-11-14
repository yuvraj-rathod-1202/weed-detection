import numpy as np

from src.gallery import CROP_RGB, HIT_RGB, render_output, shot_outcome, sugarbeets_masks


def square(x0, y0, x1, y1, size=100):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_shot_outcome_classifies_weed_crop_soil():
    weed, crop = square(10, 10, 30, 30), square(60, 60, 80, 80)
    assert shot_outcome((20, 20), [weed], [crop]) == "weed"
    assert shot_outcome((70, 70), [weed], [crop]) == "crop"
    assert shot_outcome((50, 5), [weed], [crop]) == "soil"


def test_sugarbeets_masks_splits_red_crop_from_other_colours():
    label = np.zeros((100, 100, 3), np.uint8)
    label[10:30, 10:30] = (0, 0, 255)       # sugar beet (BGR red)
    label[50:70, 50:70] = (0, 50, 255)      # weed species
    label[50:60, 5:15] = (255, 50, 0)       # another weed species
    label[80:90, 80:90] = (150, 150, 150)   # unknown: neither
    label[0:2, 95:97] = (0, 50, 255)        # 4 px speck: dropped
    weeds, crops = sugarbeets_masks(label, min_area_px=20)
    assert len(crops) == 1 and crops[0][20, 20]
    assert len(weeds) == 2 and not any(w[85, 85] for w in weeds)


def test_render_output_uses_only_allowed_colours():
    image = np.full((100, 100, 3), 128, np.uint8)
    shots = [{"x": 20, "y": 20, "outcome": "weed", "vetoed": False},
             {"x": 70, "y": 70, "outcome": "crop", "vetoed": False},
             {"x": 50, "y": 10, "outcome": "soil", "vetoed": False},
             {"x": 85, "y": 20, "outcome": "soil", "vetoed": True}]
    out = render_output(image, [square(10, 10, 30, 30)], [square(60, 60, 80, 80)], shots)
    changed = out[(out != image).any(axis=2)].astype(int)
    grey = (np.abs(changed[:, 0] - changed[:, 1]) <= 2) & (np.abs(changed[:, 1] - changed[:, 2]) <= 2)
    # translucent grey fill over grey stays grey; anything coloured must be the hit green or the crop red
    coloured = changed[~grey]
    allowed = np.array([HIT_RGB, CROP_RGB])
    assert len(coloured) > 0
    assert all(np.abs(allowed - c).sum(axis=1).min() <= 3 for c in coloured)
