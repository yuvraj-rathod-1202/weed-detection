# Notebooks

Analysis only; all pipeline logic lives in `src/`. Set the variables in the first code cell, then run all.

| Notebook | When | Shows |
|---|---|---|
| `01_data_exploration.ipynb` | after `prepare_data.py` | split/class stats, weed sizes, input → GT box → SAM mask → GT target points, which methods keep the point inside the weed, SAM failures |
| `02_model_outputs.ipynb` | after `run_experiment.py` | metrics tables, training curves, input → GT → prediction with laser points, worst cases, point-error distribution |
| `03_results_analysis.ipynb` | after downloading a run's results (no test images needed) | what 100 shots / 100 weeds turn into, overfitting check, where shots land (weed/crop species/soil), missed weeds by size and species, crop hits by species and session and why the gate missed them, confidence trade-off, aiming clearance, conclusions |

Kernel: the project `.venv` (in VS Code: "Select Kernel" → `.venv`).
