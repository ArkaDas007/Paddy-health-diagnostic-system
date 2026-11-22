# Paddy Health Diagnostic System

Combines a thermal CNN disease classifier (`predict.py`) with LightGBM quantile threshold models (`check_sensors.py` / `training_threshold.py`) to evaluate environmental sensor readings and produce a unified crop health verdict and advice.

## Project Structure

- `train_thermal.py` — Trains the thermal CNN and exports model artifacts and `predict.py` into `streamlit_app/`[cite: 3, 4, 6].
- `streamlit_app/` — Contains `predict.py`, `model_best.keras`, `model_meta.json`, and `streamlit_app.py`[cite: 6].
- `training_threshold.py` & `check_sensors.py` — Trains and runs LightGBM threshold checks using `paddy_threshold_dataset.csv` and manual/thermal health inputs[cite: 6].
- `data/` & `local_paddy_dataset/` — Thermal image dataset folders (keep ignored in Git unless using Git LFS)[cite: 3, 6].

## Setup

```bash
pip install numpy pandas opencv-python scikit-learn lightgbm tensorflow tqdm streamlit
