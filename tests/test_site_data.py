from src.site_data import SIZE_BINS, dataset_stats, size_bins


def test_size_bins_counts_values_per_bin():
    counts = size_bins([10, 20, 100])
    assert list(counts) == [label for label, _, _ in SIZE_BINS]
    assert counts["0–16"] == 1 and counts["16–24"] == 1 and counts["64–128"] == 1 and sum(counts.values()) == 3


def test_dataset_stats_counts_classes_sets_and_ignored(tmp_path):
    # rows: left, top, right, bottom, species, stem_x, stem_y; 1 = maize (crop), 31 = weed, 255 = vegetation (ignored)
    (tmp_path / "ave-0001-0001.csv").write_text("0,0,10,10,1,5,5\n0,0,20,20,31,5,5\n0,0,4,4,255,-1,-1\n")
    (tmp_path / "vwg-0002-0001.csv").write_text("0,0,30,30,31,5,5\n")
    stats = dataset_stats(tmp_path, {"train": ["ave-0001-0001"], "test": ["vwg-0002-0001"]})
    assert stats["splits"]["train"] == {"images": 1, "sessions": 1, "crop": 1, "weed": 1, "ignored": 1, "ave": 1, "vwg": 0}
    assert stats["splits"]["test"]["weed"] == 1 and stats["splits"]["test"]["vwg"] == 1
    assert stats["sizes"]["Weed"]["16–24"] == 1 and stats["sizes"]["Weed"]["24–32"] == 1
    assert stats["crop_species"] == {"Maize": 1}
