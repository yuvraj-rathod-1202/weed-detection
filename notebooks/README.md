# Notebooks

Analysis only; all pipeline logic lives in `src/`. Set the variables in the first code cell, then run all.

| Notebook | When | Shows |
|---|---|---|
| `01_data_exploration.ipynb` | after `prepare_data.py` | split/class stats, weed sizes, input → GT box → SAM mask → GT target points, which methods keep the point inside the weed, SAM failures |
| `02_model_outputs.ipynb` | after `run_experiment.py` | metrics tables, training curves, input → GT → prediction with laser points, worst cases, point-error distribution |

Kernel: the project `.venv` (in VS Code: "Select Kernel" → `.venv`).
