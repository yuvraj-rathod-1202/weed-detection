import numpy as np

from src.baseline import baseline_prediction, vegetation_mask

SOIL_BGR = (60, 90, 120)    # brown
LEAF_BGR = (40, 160, 50)    # green


def _image(squares, size=100):
    image = np.full((size, size, 3), SOIL_BGR, np.uint8)
    for x0, y0, x1, y1 in squares:
        image[y0:y1, x0:x1] = LEAF_BGR
    return image


def test_vegetation_mask_finds_green_on_soil():
    mask = vegetation_mask(_image([(40, 40, 60, 60)]))
    truth = np.zeros((100, 100), bool)
    truth[40:60, 40:60] = True
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou > 0.9


def test_baseline_drops_small_specks():
    pred = baseline_prediction(_image([(5, 5, 10, 10), (40, 40, 60, 60)]))
    assert len(pred.masks) == 1 and pred.confidences == [1.0] and pred.class_ids == [1]
    assert pred.masks[0][50, 50] and not pred.masks[0][7, 7]


def test_baseline_on_bare_soil_predicts_nothing():
    pred = baseline_prediction(_image([]))
    assert pred.masks == []
