import pytest

from src.evaluation import crop_class_ids, split_crops


def test_crop_class_ids_maps_names_to_ids():
    assert crop_class_ids(["Waterhemp", "Cotton", "Purslane"], ["Cotton"]) == {1}
    assert crop_class_ids(["Waterhemp"], []) == set()


def test_unknown_crop_class_raises():
    with pytest.raises(ValueError, match="cotton"):
        crop_class_ids(["Waterhemp", "Cotton"], ["cotton"])


def test_split_crops_keeps_order():
    assert split_crops([0, 1, 2, 1], {1}) == ([0, 2], [1, 3])
