import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "03_results_analysis.ipynb"


def test_results_notebook_embeds_no_dataset_photos():
    """CropAndWeed's licence forbids distributing its images, also as overlays in notebook outputs."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert not re.search(r"(imread|imshow)\([^\n]*vis", source), source[:200]
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            assert "image/png" not in data and "image/jpeg" not in data
