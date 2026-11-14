from src.selection import select_final


def _summary(hit, ms=50.0, threshold=0.3):
    optimal = None if hit is None else {"threshold": threshold, "weed_hit_rate": hit, "crop_shot_share": 0.01, "shots": 10}
    return {"optimal": optimal, "inference_ms_per_image": ms}


def test_select_final_picks_highest_hit_rate_within_limit():
    chosen = select_final({"a": _summary(0.80), "b": _summary(0.85, threshold=0.2), "c": _summary(None)})
    assert chosen["run"] == "b" and chosen["threshold"] == 0.2 and chosen["val"]["weed_hit_rate"] == 0.85


def test_select_final_breaks_ties_by_speed():
    assert select_final({"slow": _summary(0.8, ms=80), "fast": _summary(0.8, ms=40)})["run"] == "fast"


def test_select_final_none_when_no_candidate_qualifies():
    assert select_final({"a": _summary(None), "b": _summary(None)}) is None
