# COMP8410 Assignment 2 Reproducibility Repository

This repository contains the code, input data, generated outputs, and submitted report artifacts for Assignment 2. It is organised so that a marker can either rerun the analysis from source or inspect the already-generated outputs directly.

## Repository Structure

- `assignment2.py`: main analysis script.
- `assignment2_distributed/`: assignment brief, input CSV, data dictionary, and source materials distributed with the assignment.
- `assignment2_outputs/`: generated tables, figures, notes, and run summary produced by the script.


## Environment

The script was written for Python 3 and uses the following libraries:

- `numpy`
- `pandas`
- `Pillow`
- `scikit-learn`

Example installation:

```bash
pip install numpy pandas pillow scikit-learn
```

## How To Reproduce

Run the main script from the repository root:

```bash
python assignment2.py
```

The script expects these input files to already exist:

- `assignment2_distributed/Indigenous_Voters_pp.csv`
- `assignment2_distributed/1_IndigenousVoters_2025_Data_Dictionary_100297_GENERAL.xlsx`

## Outputs

Running the script writes outputs to `assignment2_outputs/`, including:

- classification summary tables and confusion matrices
- feature importance tables and tree text
- association rule tables
- clustering validity and cluster profile tables
- generated figures under `assignment2_outputs/figures/`
- `corrected_run_summary.json`
- `corrected_report_notes.md`

The script also tries to refresh figures inside `Indigenous_Voters_Data_Mining_Report_updated.docx` if that file is present. If the document is missing, the analysis still runs and the report-refresh step is skipped.

## Reproducibility Notes

- Cross-validation is configured with a fixed random seed (`random_state=8410`).
- K-means clustering is also run with a fixed random seed (`random_state=8410`).
- The repository already includes generated outputs, so rerunning the script is optional for inspection purposes.

## Quick Marker Guide

If you do not want to rerun the analysis, the main files to inspect are:

- `assignment2.py`
- `assignment2_outputs/corrected_run_summary.json`
- `assignment2_outputs/corrected_report_notes.md`
- the CSV tables and figures inside `assignment2_outputs/`
