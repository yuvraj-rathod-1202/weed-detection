"""Choose the final model on the validation split with one fixed rule.

Each candidate's validation summary (src/split_eval.py) holds its optimal threshold: the most weeds hit
while at most 2 % of shots land on crops. The final model is the candidate whose optimum hits the most
weeds (ties: the faster one). Candidates with no threshold inside the crop limit are out. The baseline
is never a candidate. Only the selected model is then evaluated on test.
"""
from __future__ import annotations


def select_final(summaries: dict[str, dict]) -> dict | None:
    """{run name: summary.json} -> {"run", "threshold", "val": optimal row}, or None if nobody qualifies."""
    ok = {run: s for run, s in summaries.items() if s.get("optimal")}
    if not ok:
        return None
    run = max(ok, key=lambda r: (ok[r]["optimal"]["weed_hit_rate"], -(ok[r]["inference_ms_per_image"] or 0.0)))
    return {"run": run, "threshold": ok[run]["optimal"]["threshold"], "val": ok[run]["optimal"]}
