# Paddy Health Diagnostic System

Combines a thermal CNN disease classifier (`predict.py`) with LightGBM quantile threshold models (`check_sensors.py` / `training_threshold.py`) to evaluate environmental sensor readings and produce a unified crop health verdict and advice.

## Project Structure

- `train_thermal.py` — Trains the thermal CNN and exports model artifacts and `predict.py` into `streamlit_app/`.
- `streamlit_app/` — Contains `predict.py`, `model_best.keras`, `model_meta.json`, and `streamlit_app.py`.
- `training_threshold.py` & `check_sensors.py` — Trains and runs LightGBM threshold checks using `paddy_threshold_dataset.csv` and manual/thermal health inputs.

## Setup

```bash
pip install numpy pandas opencv-python scikit-learn lightgbm tensorflow tqdm streamlit
