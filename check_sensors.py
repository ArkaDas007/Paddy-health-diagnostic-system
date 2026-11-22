import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
STREAMLIT_DIR = BASE_DIR / "streamlit_app"
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

try:
    from predict import predict as predict_thermal_image
except Exception as e:
    predict_thermal_image = None
    print(f"[warning] Thermal predict module not loaded: {e}")

DATASET_PATH = str(BASE_DIR / "paddy_threshold_dataset.csv")

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

THRESHOLD_TO_SENSOR = {
    "min_temp_c": ("dht22_temp_c", "DHT22 Temp"),
    "min_humidity_pct": ("dht22_humidity_pct", "DHT22 Humidity"),
    "min_moisture_pct": ("soil_moisture_pct", "Soil Moisture"),
    "min_light_lux": ("bh1750_lux", "BH1750 Light"),
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

VALID_STAGES = ["seedling", "tillering", "flowering", "ripening"]
VALID_STATES = ["Healthy", "Diseased", "Water-Stressed", "Nutrient-deficient"]


def train_models(csv_path: str = DATASET_PATH, quantile: float = 0.2) -> dict[str, Pipeline]:
    df = pd.read_csv(csv_path)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    models: dict[str, Pipeline] = {}
    for target in THRESHOLD_TARGETS:
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=quantile,
                        n_estimators=200,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X, df[target])
        models[target] = pipe

    return models


def prompt_choice(label: str, choices: list[str]) -> str:
    options_str = ", ".join(choices)
    while True:
        val = input(f"{label} ({options_str}): ").strip()
        for c in choices:
            if val.lower() == c.lower():
                return c
        print(f"  Invalid input. Choose one of: {options_str}")


def prompt_float(label: str, min_val: float | None = None, max_val: float | None = None) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            num = float(raw)
            if min_val is not None and num < min_val:
                print(f"  Value must be >= {min_val}.")
                continue
            if max_val is not None and num > max_val:
                print(f"  Value must be <= {max_val}.")
                continue
            return num
        except ValueError:
            print("  Please enter a valid number.")


def resolve_health_state() -> str:
    if predict_thermal_image is not None:
        use_img = input("Predict health state from thermal image? (y/n): ").strip().lower()
        if use_img == "y":
            img_path = input("Enter thermal image path: ").strip().strip('"').strip("'")
            try:
                res = predict_thermal_image(img_path)
                detected = str(res["label"])
                conf = float(res["confidence"])
                prob_dis = float(res["prob_diseased"])
                print(
                    f"  Thermal CNN -> {detected} "
                    f"(confidence: {conf:.2%}, P(diseased): {prob_dis:.2%})"
                )
                if detected in VALID_STATES:
                    return detected
                print(
                    f"  [note] Label '{detected}' not in {VALID_STATES}; "
                    "choosing health state manually."
                )
            except Exception as e:
                print(f"  [warning] Thermal prediction failed ({e}); choosing manually.")

    return prompt_choice("Health state", VALID_STATES)


def get_manual_input() -> pd.DataFrame:
    print("\n--- Enter Context ---")
    growth_stage = prompt_choice("Growth stage", VALID_STAGES)
    health_state = resolve_health_state()
    hour_of_day = prompt_float("Hour of day (0-24, e.g. 10.5)", min_val=0.0, max_val=24.0)

    print("\n--- Enter Manual Sensor Readings ---")
    dht22_temp_c = prompt_float("DHT22 Temperature (°C)")
    dht22_humidity_pct = prompt_float("DHT22 Humidity (%)", min_val=0.0, max_val=100.0)
    ds18b20_temp_c = prompt_float("DS18B20 Probe Temperature (°C)")
    soil_moisture_pct = prompt_float("Capacitive Soil Moisture (%)", min_val=0.0, max_val=100.0)
    bh1750_lux = prompt_float("BH1750 Light Intensity (lux)", min_val=0.0)

    return pd.DataFrame(
        [
            {
                "growth_stage": growth_stage,
                "health_state": health_state,
                "dht22_temp_c": dht22_temp_c,
                "dht22_humidity_pct": dht22_humidity_pct,
                "ds18b20_temp_c": ds18b20_temp_c,
                "soil_moisture_pct": soil_moisture_pct,
                "bh1750_lux": bh1750_lux,
                "hour_of_day": hour_of_day,
            }
        ]
    )


def build_advice(below_min_labels: list[str], health_state: str) -> str:
    tips: list[str] = []

    moisture_low = "Soil Moisture" in below_min_labels
    humidity_low = "DHT22 Humidity" in below_min_labels
    temp_low = "DHT22 Temp" in below_min_labels
    light_low = "BH1750 Light" in below_min_labels

    if health_state == "Water-Stressed":
        if moisture_low or humidity_low:
            tips.append("Add water / irrigate soon and check soil moisture and dry air/humidity.")
        else:
            tips.append(
                "Moisture and humidity are at or above minimum, so check irrigation distribution, "
                "root zone, or heat/transpiration stress."
            )
    elif health_state == "Nutrient-deficient":
        tips.append(
            "Review nutrient/fertilizer schedule (e.g., nitrogen/potassium/zinc) and verify "
            "soil moisture is adequate for uptake."
        )
    elif health_state == "Diseased":
        tips.append(
            "Inspect leaves/panicles for disease symptoms and manage excess moisture/humidity "
            "that favors fungal or bacterial spread."
        )
    elif health_state == "Healthy" and not below_min_labels:
        tips.append("Conditions look good—maintain current watering and care routine.")

    if moisture_low and health_state != "Water-Stressed":
        tips.append("Soil moisture is below minimum—consider adding water.")
    if humidity_low and not moisture_low and health_state != "Water-Stressed":
        tips.append("Humidity is below minimum—monitor canopy drying and heat stress.")
    if temp_low:
        tips.append("Temperature is below minimum—watch for cool-temperature slowdown.")
    if light_low:
        tips.append("Light level is below minimum for this time/stage—check shading or cloud cover.")

    if not tips:
        tips.append("Monitor crop condition and re-check sensor readings.")

    # Deduplicate while preserving order
    return " ".join(dict.fromkeys(tips))


def build_verdict(results: list[dict], health_state: str, ds18b20_temp_c: float) -> str:
    below_min = [r["label"] for r in results if r["status"] == "Below Min"]

    if not below_min:
        env_status = "Environment Good (All checked factors >= Min Required)"
    else:
        env_status = f"Action Needed (Below Min: {', '.join(below_min)})"

    if not below_min and health_state == "Healthy":
        summary = "Good — Minimum thresholds met and plant health is Healthy."
    elif below_min:
        summary = (
            f"Attention Required — {', '.join(below_min)} is below minimum "
            f"(Current health state: {health_state}; DS18B20 probe: {ds18b20_temp_c:.2f} °C)."
        )
    else:
        summary = (
            f"Monitor Plant — Environment meets minimums, but health state is {health_state}."
        )

    advice = build_advice(below_min, health_state)

    return (
        "\n--- FINAL VERDICT ---\n"
        f"Environmental Status : {env_status}\n"
        f"Plant Health Status  : {health_state}\n"
        f"Overall Verdict      : {summary}\n"
        f"Advice               : {advice}"
    )


def main() -> None:
    models = train_models(DATASET_PATH)

    while True:
        sample = get_manual_input()
        row = sample.iloc[0]

        results = []
        print("\n--- Threshold Check Results ---")
        for target, model in models.items():
            min_required = float(model.predict(sample)[0])
            unit = TARGET_UNITS[target]
            sensor_col, sensor_label = THRESHOLD_TO_SENSOR[target]
            actual_val = float(row[sensor_col])
            status = "Good" if actual_val >= min_required else "Below Min"
            results.append(
                {
                    "label": sensor_label,
                    "actual": actual_val,
                    "min_required": min_required,
                    "unit": unit,
                    "status": status,
                }
            )
            print(
                f"  {sensor_label:15s}: {actual_val:8.2f} {unit:4s} | "
                f"Min Required: {min_required:8.2f} {unit:4s} | {status}"
            )

        ds18b20_val = float(row["ds18b20_temp_c"])
        print(f"  {'DS18B20 Probe':15s}: {ds18b20_val:8.2f} {'°C':4s} | Reference reading logged")

        print(build_verdict(results, str(row["health_state"]), ds18b20_val))

        again = input("\nCheck another reading? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()