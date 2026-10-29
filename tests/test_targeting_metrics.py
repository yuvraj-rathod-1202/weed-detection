import numpy as np
import pytest

from src.targeting.metrics import aggregate_targeting, evaluate_image, mask_iou, match_instances


def rect(x0, y0, x1, y1, size=100):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_mask_iou():
    assert mask_iou(rect(0, 0, 10, 10), rect(0, 0, 10, 10)) == 1.0
    assert mask_iou(rect(0, 0, 10, 10), rect(50, 50, 60, 60)) == 0.0
    assert mask_iou(rect(0, 0, 10, 10), rect(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_match_is_one_to_one_and_confidence_ordered():
    gt = [rect(0, 0, 20, 20)]
    preds = [rect(0, 0, 20, 20), rect(1, 1, 20, 20)]  # both overlap the single GT
    assert [(p, g) for p, g, _ in match_instances(preds, gt, 0.5)] == [(0, 0)]


def test_perfect_prediction():
    gt = [rect(10, 10, 40, 40), rect(60, 60, 90, 90)]
    record = evaluate_image(list(gt), gt, ["distance_transform", "box_center"], match_iou=0.5)
    summary = aggregate_targeting([record], ["distance_transform", "box_center"])
    for method in ("distance_transform", "box_center"):
        m = summary["methods"][method]
        assert m["shot_precision"] == 1.0
        assert m["weed_hit_rate"] == 1.0
        assert m["point_error_px"]["mean"] == 0.0


def test_false_positive_and_missed_weed():
    gt = [rect(10, 10, 40, 40), rect(60, 60, 90, 90)]
    preds = [rect(10, 10, 40, 40), rect(0, 60, 5, 65)]  # one correct, one on soil; second weed missed
    record = evaluate_image(preds, gt, ["centroid"], match_iou=0.5)
    m = aggregate_targeting([record], ["centroid"])["methods"]["centroid"]
    assert m["shot_precision"] == 0.5
    assert m["weed_hit_rate"] == 0.5
    assert m["point_error_px"]["n"] == 1


def test_no_predictions_gives_undefined_precision():
    record = evaluate_image([], [rect(10, 10, 40, 40)], ["centroid"], match_iou=0.5)
    m = aggregate_targeting([record], ["centroid"])["methods"]["centroid"]
    assert m["shot_precision"] is None
    assert m["weed_hit_rate"] == 0.0


def test_clearance_and_center_error():
    gt = np.zeros((100, 100), bool)
    gt[20:80, 20:80] = True
    rec = evaluate_image([gt.copy()], [gt], ["centroid"], 0.5)["methods"]["centroid"]
    assert rec["clearance_px"][0] >= 29  # centre of a 60 px square: ~30 px from every edge
    assert rec["center_errors_rel"][0] < 0.05


def test_without_crop_labels_crop_metrics_are_undefined():
    gt = [rect(10, 10, 40, 40)]
    m = aggregate_targeting([evaluate_image(list(gt), gt, ["centroid"], 0.5)], ["centroid"])["methods"]["centroid"]
    assert m["crop_hit_rate"] is None
    assert m["veto_rate"] == 0.0


def test_shot_on_crop_counts_as_crop_hit():
    weed, crop = rect(10, 10, 40, 40), rect(60, 60, 90, 90)
    preds = [weed, crop]  # the model mistakes the crop for a weed and finds no crop
    record = evaluate_image(preds, [weed], ["centroid"], 0.5, gt_crop_masks=[crop])
    m = aggregate_targeting([record], ["centroid"])["methods"]["centroid"]
    assert m["crop_hit_rate"] == 0.5
    assert m["crop_hit_rate_ungated"] == 0.5
    assert m["shot_precision"] == 0.5


def test_gate_vetoes_shot_near_predicted_crop():
    weed, crop = rect(10, 10, 40, 40), rect(60, 60, 90, 90)
    preds = [weed, crop]
    record = evaluate_image(preds, [weed], ["centroid"], 0.5, pred_crop_masks=[crop], gt_crop_masks=[crop],
                            crop_margin_px=5)
    m = aggregate_targeting([record], ["centroid"])["methods"]["centroid"]
    assert m["crop_hit_rate"] == 0.0         # the bad shot was not fired
    assert m["crop_hit_rate_ungated"] == 0.5  # ...but would have hit the crop without the gate
    assert m["veto_rate"] == 0.5
    assert m["shot_precision"] == 1.0
    assert m["weed_hit_rate"] == 1.0


def test_gate_costs_recall_when_weed_touches_crop():
    weed, crop = rect(10, 10, 40, 40), rect(40, 10, 70, 40)  # weed grows right against the crop
    record = evaluate_image([weed], [weed], ["centroid"], 0.5, pred_crop_masks=[crop], gt_crop_masks=[crop],
                            crop_margin_px=20)
    m = aggregate_targeting([record], ["centroid"])["methods"]["centroid"]
    assert m["weed_hit_rate"] == 0.0  # centroid is ~15 px from the crop: too close, not fired
    assert m["veto_rate"] == 1.0
