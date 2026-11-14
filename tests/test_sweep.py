import numpy as np

from src.targeting.sweep import bootstrap_ci, fired, optimal_threshold, shot_records, sweep


def square(x0, y0, x1, y1, size=100):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def shot(conf, crop=0.0, ignored=False, gt_weed=0, on_crop=False):
    return {"conf": conf, "x": 50.0, "y": 50.0, "ignored": ignored, "gt_weed": gt_weed, "on_crop": on_crop,
            "max_nearby_crop_conf": crop}


def test_shot_hits_weed_and_records_nearby_crop_conf():
    weed = square(40, 40, 60, 60)
    crop = square(66, 40, 86, 60)  # 6 px right of the weed's edge, ~16 px from its target point
    near = shot_records([weed], [0.8], [crop], [0.3], [weed], [crop], ignored=set(), margin_px=20)
    assert len(near) == 1
    assert near[0]["gt_weed"] == 0 and near[0]["on_crop"] is False
    assert near[0]["max_nearby_crop_conf"] == 0.3
    far = shot_records([weed], [0.8], [crop], [0.3], [weed], [crop], ignored=set(), margin_px=5)
    assert far[0]["max_nearby_crop_conf"] == 0.0


def test_fired_applies_threshold_ignore_and_gate():
    shots = [shot(0.4), shot(0.4, crop=0.5), shot(0.2), shot(0.9, ignored=True)]
    assert fired(shots, 0.3) == [shots[0]]
    assert fired(shots, 0.6) == []


def test_sweep_rates_and_empty_images():
    records = [
        {"image_id": "ave-0001-0001", "gt_weeds": [{"size_px": 20}, {"size_px": 30}],
         "shots": [shot(0.9, gt_weed=0), shot(0.9, gt_weed=-1, on_crop=True)]},
        {"image_id": "ave-0002-0001", "gt_weeds": [], "shots": []},
    ]
    row = sweep(records, [0.05])[0]
    assert row["shots"] == 2
    assert row["weed_hit_rate"] == 0.5 and row["crop_shot_share"] == 0.5
    assert row["shot_precision"] == 0.5 and row["soil_shot_share"] == 0.0
    empty = sweep([], [0.05])[0]
    assert empty["shots"] == 0 and empty["weed_hit_rate"] is None and empty["crop_shot_share"] is None


def test_optimal_threshold_respects_crop_limit():
    curve = [
        {"threshold": 0.1, "shots": 10, "weed_hit_rate": 0.9, "crop_shot_share": 0.05},
        {"threshold": 0.3, "shots": 8, "weed_hit_rate": 0.8, "crop_shot_share": 0.02},
        {"threshold": 0.5, "shots": 5, "weed_hit_rate": 0.7, "crop_shot_share": 0.0},
    ]
    assert optimal_threshold(curve)["threshold"] == 0.3
    assert optimal_threshold([dict(r, crop_shot_share=0.05) for r in curve]) is None


def test_empty_mask_gives_no_shot():
    empty = np.zeros((100, 100), bool)
    assert shot_records([empty], [0.9], [], [], [], [], ignored=set(), margin_px=20) == []


def test_bootstrap_ci_brackets_point_estimate():
    records = [{"image_id": f"vwg-{i:04d}-0001", "gt_weeds": [{"size_px": 20}],
                "shots": [shot(0.9, gt_weed=0)] if i < 6 else []} for i in range(10)]
    ci = bootstrap_ci(records, 0.5, n=500)
    lo, hi = ci["weed_hit_rate"]
    assert lo <= 0.6 <= hi and lo < hi
