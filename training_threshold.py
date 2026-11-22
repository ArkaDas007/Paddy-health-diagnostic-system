import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

DATASET_PATH = "paddy_threshold_dataset.csv"

THRESHOLD_TARGETS = [
    "min_temp_c",
    "min_humidity_pct",
    "min_moisture_pct",
    "min_light_lux",
]

TARGET_UNITS = {
    "min_temp_c": "°C",
    "min_humidity_pct": "%",
    "min_moisture_pct": "%",
    "min_light_lux": "lux",
}

CATEGORICAL_FEATURES = ["growth_stage", "health_state"]
NUMERIC_FEATURES = [
    "dht22_temp_c",
    "dht22_humidity_pct",
    "ds18b20_temp_c",
    "soil_moisture_pct",
    "bh1750_lux",
    "hour_of_day",
]


def load_paddy_dataset(csv_path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + THRESHOLD_TARGETS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return df

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

def train_lightgbm_threshold_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    quantile: float = 0.2,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for target in THRESHOLD_TARGETS:
        regressor = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("model", regressor),
            ]
        )
        pipe.fit(X_train, y_train[target])
        models[target] = pipe
    return models

def train_xgboost_threshold_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    quantile: float = 0.2,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for target in THRESHOLD_TARGETS:
        regressor = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            tree_method="hist",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("model", regressor),
            ]
        )
        pipe.fit(X_train, y_train[target])
        models[target] = pipe
    return models

def evaluate_models(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    quantile: float = 0.2,
    model_name: str = "LightGBM",
) -> pd.DataFrame:
    rows = []
    for target, pipe in models.items():
        y_true = y_test[target].to_numpy()
        y_pred = pipe.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "target": target,
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": root_mean_squared_error(y_true, y_pred),
                f"pinball_q{int(quantile * 100)}": mean_pinball_loss(
                    y_true, y_pred, alpha=quantile
                ),
            }
        )
    return pd.DataFrame(rows)

def main() -> None:
    df = load_paddy_dataset(DATASET_PATH)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df[THRESHOLD_TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    quantile = 0.2
    lgb_models = train_lightgbm_threshold_models(X_train, y_train, quantile=quantile)
    xgb_models = train_xgboost_threshold_models(X_train, y_train, quantile=quantile)

    lgb_metrics = evaluate_models(
        lgb_models, X_test, y_test, quantile=quantile, model_name="LightGBM"
    )
    xgb_metrics = evaluate_models(
        xgb_models, X_test, y_test, quantile=quantile, model_name="XGBoost"
    )

    results = pd.concat([lgb_metrics, xgb_metrics], ignore_index=True)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    sample_reading = pd.DataFrame(
        [
            {
                "growth_stage": "tillering",
                "health_state": "Water-Stressed",
                "dht22_temp_c": 31.5,
                "dht22_humidity_pct": 64.0,
                "ds18b20_temp_c": 29.8,
                "soil_moisture_pct": 48.0,
                "bh1750_lux": 24000.0,
                "hour_of_day": 11.5,
            }
        ]
    )
    predicted_thresholds = {
        target: float(model.predict(sample_reading)[0])
        for target, model in lgb_models.items()
    }
    print("\nPredicted minimum thresholds for sample reading:")
    for target, model in lgb_models.items():
        value = float(model.predict(sample_reading)[0])
        unit = TARGET_UNITS[target]
        print(f"  {target}: {value:.2f} {unit}")

if __name__ == "__main__":
    main()