from src.site_data import SIZE_BINS, dataset_stats, hit_rate_by, paired_bootstrap_diff, size_bins


def _shot(conf, gt_weeds=(), on_crop=False):
    return {"conf": conf, "ignored": False, "gt_weed": gt_weeds[0] if gt_weeds else -1, "gt_weeds": list(gt_weeds),
            "gt_crop": 0 if on_crop else -1, "on_crop": on_crop, "max_nearby_crop_conf": 0.0}


def test_hit_rate_by_groups_weeds_and_counts_hits_at_threshold():
    records = [
        {"image_id": "ave-0001-0001", "gt_weeds": [{"size_px": 20}, {"size_px": 100}], "shots": [_shot(0.9, [1])]},
        {"image_id": "vwg-0002-0001", "gt_weeds": [{"size_px": 30}], "shots": [_shot(0.3, [0])]},
    ]
    by_set = hit_rate_by(records, 0.5, lambda record, weed: record["image_id"][:3])
    assert by_set == {"ave": {"weeds": 2, "hit": 1, "rate": 0.5}, "vwg": {"weeds": 1, "hit": 0, "rate": 0.0}}
    by_size = hit_rate_by(records, 0.2, lambda record, weed: "small" if weed["size_px"] <= 24 else "large")
    assert by_size["small"]["rate"] == 0.0 and by_size["large"]["rate"] == 1.0


def test_public_record_drops_points_and_exact_sizes_but_keeps_what_the_site_needs():
    from src.site_data import public_record

    record = {"image_id": "ave-0001-0001", "session": "ave-0001", "inference_ms": 40.0, "num_gt_crops": 1,
              "gt_weeds": [{"size_px": 20.3}, {"size_px": 100.0}],
              "shots": [{**_shot(0.9, [1]), "x": 512.0, "y": 300.0}]}
    pub = public_record(record)
    assert pub["gt_weeds"] == [{"size_bin": "16–24"}, {"size_bin": "64–128"}]
    assert "x" not in pub["shots"][0] and "y" not in pub["shots"][0]
    assert pub["shots"][0]["gt_weeds"] == [1] and pub["num_gt_crops"] == 1
    by_size = hit_rate_by([pub], 0.5, lambda r, w: w["size_bin"])
    assert by_size["64–128"]["rate"] == 1.0 and by_size["16–24"]["rate"] == 0.0


def test_paired_bootstrap_diff_is_zero_for_identical_runs_and_positive_for_better():
    records = [{"image_id": f"vwg-{i:04d}-0001", "gt_weeds": [{"size_px": 30}], "shots": [_shot(0.9, [0])] if i % 2 else []}
               for i in range(20)]
    better = [dict(r, shots=[_shot(0.9, [0])]) for r in records]
    same = paired_bootstrap_diff(records, records, 0.5, 0.5, n=300)
    assert same["diff"] == 0.0 and same["ci"] == [0.0, 0.0]
    up = paired_bootstrap_diff(better, records, 0.5, 0.5, n=300)
    assert up["diff"] == 0.5 and up["ci"][0] > 0


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
