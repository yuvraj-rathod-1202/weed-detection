import numpy as np
import pytest

from src.targeting.points import POINT_METHODS, point_in_mask, target_point


def square_mask(size=100, x0=20, x1=60, y0=30, y1=70):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def c_shape_mask():
    """A 'C' whose centroid lies in the empty gap, outside the mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:30] = True   # left bar
    mask[10:30, 10:90] = True   # top bar
    mask[70:90, 10:90] = True   # bottom bar
    return mask


@pytest.mark.parametrize("method", sorted(POINT_METHODS))
def test_square_point_is_near_center(method):
    x, y = target_point(square_mask(), method)
    assert abs(x - 39.5) <= 2 and abs(y - 49.5) <= 2


@pytest.mark.parametrize("method", sorted(POINT_METHODS))
def test_empty_mask_returns_none(method):
    assert target_point(np.zeros((50, 50), dtype=bool), method) is None


def test_centroid_can_miss_non_convex_weed_but_distance_transform_cannot():
    mask = c_shape_mask()
    assert not point_in_mask(target_point(mask, "centroid"), mask)
    assert point_in_mask(target_point(mask, "distance_transform"), mask)
    assert point_in_mask(target_point(mask, "skeleton"), mask)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        target_point(square_mask(), "nope")


def test_point_in_mask_bounds():
    mask = square_mask()
    assert point_in_mask((40, 50), mask)
    assert not point_in_mask((5, 5), mask)
    assert not point_in_mask((500, 500), mask)
    assert not point_in_mask(None, mask)
