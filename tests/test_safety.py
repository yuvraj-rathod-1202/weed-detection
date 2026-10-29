import numpy as np

from src.targeting.safety import vetoed_shots


def rect(x0, y0, x1, y1, size=100):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_no_crop_vetoes_nothing():
    assert vetoed_shots([(10.0, 10.0), (50.0, 50.0)], [], margin_px=20) == []


def test_shot_inside_crop_is_vetoed_even_without_margin():
    crop = [rect(40, 40, 60, 60)]
    assert vetoed_shots([(50.0, 50.0), (10.0, 10.0)], crop, margin_px=0) == [0]


def test_margin_vetoes_shots_close_to_crop():
    crop = [rect(40, 40, 60, 60)]
    near, far = (35.0, 50.0), (20.0, 50.0)  # 5 px and 20 px left of the crop
    assert vetoed_shots([near, far], crop, margin_px=10) == [0]


def test_missing_points_are_not_vetoed():
    assert vetoed_shots([None, (50.0, 50.0)], [rect(40, 40, 60, 60)], margin_px=0) == [1]
