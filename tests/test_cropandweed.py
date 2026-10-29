import numpy as np
import pytest

from src.data.cropandweed import CLASS_NAMES, convert_rows, convert_semantic, map_label


def test_official_crop_or_weed2_mapping():
    assert CLASS_NAMES == ["Crop", "Weed"]
    assert map_label(1) == 0 and map_label(94) == 0     # maize, soybean
    assert map_label(48) == 1 and map_label(79) == 1    # meadow-grass, chickweed
    assert map_label(14) is None and map_label(40) is None  # courgette, purslane: not in CropOrWeed2
    assert map_label(255) is None                        # "Vegetation"


def test_convert_rows_splits_targets_and_ignore_regions():
    rows = [
        (100, 100, 300, 200, 1, 200, 150),   # maize -> Crop box with stem
        (0, 0, 100, 100, 255, -1, -1),       # Vegetation -> ignore region
        (500, 500, 600, 700, 48, 550, 600),  # meadow-grass -> Weed box
    ]
    boxes, ignore, stems = convert_rows(rows, width=1000, height=1000)
    assert boxes == [pytest.approx((0, 0.2, 0.15, 0.2, 0.1)), pytest.approx((1, 0.55, 0.6, 0.1, 0.2))]
    assert ignore == [pytest.approx((0.05, 0.05, 0.1, 0.1))]
    assert stems == [pytest.approx((0.2, 0.15)), pytest.approx((0.55, 0.6))]


def test_missing_stem_is_none():
    _, _, stems = convert_rows([(10, 10, 20, 20, 1, -1, -1)], width=100, height=100)
    assert stems == [None]


def test_convert_semantic_uses_class_id_plus_one():
    source = np.array([[0, 1, 94], [48, 40, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(convert_semantic(source), [[0, 1, 1], [2, 255, 255]])


def test_convert_dataset_writes_the_raw_layout_prepare_reads(tmp_path):
    import cv2

    from src.data.cropandweed import convert_dataset
    from src.data.label_fixes import read_ignore_regions
    from src.data.labels import read_class_names, read_yolo_boxes

    src, out = tmp_path / "src", tmp_path / "raw"
    for d in ("images", "bboxes", "labelIds/CropAndWeed"):
        (src / d).mkdir(parents=True)
    cv2.imwrite(str(src / "images" / "vwg-0001-0002.jpg"), np.zeros((100, 200, 3), np.uint8))
    mask = np.zeros((100, 200), np.uint8)
    mask[10:30, 10:30] = 1
    cv2.imwrite(str(src / "labelIds" / "CropAndWeed" / "vwg-0001-0002.png"), mask)
    (src / "bboxes" / "vwg-0001-0002.csv").write_text("10,10,30,30,1,20,20\n50,50,60,60,255,55,55\n")
    cv2.imwrite(str(src / "images" / "vwg-0001-0003.jpg"), np.zeros((100, 200, 3), np.uint8))  # no annotations

    assert convert_dataset(src, out) == {"images": 1, "converted_now": 1}
    assert read_class_names(out / "classes.txt") == ["Crop", "Weed"]
    assert read_yolo_boxes(out / "labels" / "vwg-0001-0002.txt") == [pytest.approx((0, 0.1, 0.2, 0.1, 0.2))]
    assert read_ignore_regions(out / "ignore" / "vwg-0001-0002.txt") == [pytest.approx((0.275, 0.55, 0.05, 0.1))]
    assert cv2.imread(str(out / "semantic" / "vwg-0001-0002.png"), cv2.IMREAD_GRAYSCALE)[20, 20] == 1
    assert (out / "stems" / "vwg-0001-0002.txt").read_text() == "0.100000 0.200000\n"
    assert convert_dataset(src, out)["converted_now"] == 0  # resumable
