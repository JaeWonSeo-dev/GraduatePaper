# GraduatePaper_Final

Cross-domain IDS experiment project for evaluating the proposed method (**FN-Focal Loss + Attack Pattern Head**) against baseline variants under the same preprocessing and training conditions.

## Project goal

- **Train/validation**: CICIDS-2017
- **Test**: UNSW-NB15
- **Task**: Binary intrusion detection (Normal=0, Attack=1)
- **Main comparison**: Baseline vs Proposed method across fixed model combinations

The focus of this repository is code and reproducibility guidance. Large raw datasets are intentionally excluded from Git.

## Main files

### Core code
- `runner_combine.py`: main experiment runner
- `fn_focal_attackhead.py`: FN-Focal Loss and AP-Head related logic
- `feature_mapping.py`: feature alignment / mapping helpers
- `run_new_experiments.py`: helper script for running the newer experiment set

### Supporting scripts
- `check_dataset_distribution.py`
- `check_experiment_params.py`
- `verify_critical_fixes.py`
- `test_mapping.py`
- `nb15_add_headers.py`
- `feature_analysis.py`

## Included documentation

Only the documents needed to understand and run the project are tracked for GitHub:

- `README.md`: repository overview and setup guidance
- `EXPERIMENT_GUIDE_NEW.md`: execution guide for the current experiment design

Patch logs, intermediate change notes, validation checklists, and drafting notes are intentionally omitted from the initial Git history to keep the repository focused.

## Dataset layout (not committed)

Expected local paths:

- `CSV/MachineLearningCSV/MachineLearningCVE/` for CICIDS-2017
- `CSV_NB15/CSV Files/Training and Testing Sets/` for UNSW-NB15

These directories are excluded by `.gitignore` because they are too large for a normal Git repository.

## Recommended environment

Create a clean Python environment and install the dependencies used by the project code. If you want, a dedicated `requirements.txt` or `environment.yml` can be added later.

## Typical usage

Run the current experiment design:

```powershell
python runner_combine.py
```

Or run the scripted experiment batch:

```powershell
python run_new_experiments.py
```

For the detailed experiment flow, parameter meaning, and output expectations, see `EXPERIMENT_GUIDE_NEW.md`.

## Outputs

Typical generated outputs:

- `results_run.csv`
- confusion matrix / run artifacts under `runs/`

These are excluded from Git because they are generated results, not source files.

## Notes

- This repository is organized for publication and review.
- Large datasets, virtual environments, logs, installers, and temporary outputs are excluded.
- If needed, the repository can be cleaned further into a more formal layout (`src/`, `docs/`, `requirements.txt`).
