
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("======================================")
print(" DAILY ELECTRICITY CONSUMPTION FORECAST")
print("======================================")

# 1. Project paths
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "model" / "best_tuned_model.pkl"

DATA_PATH = (
    ROOT / "dataset" / "processed"
    / "electricity_features.csv"
)

OUTPUT_PATH = (
    ROOT / "prediction"
    / "daily_electricity_predictions.csv"
)

# 2. Check files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# 3. Load trained model
print("\n1. Loading trained model...")

package = joblib.load(MODEL_PATH)

model = package["model"]
features = package["features"]
needs_scaling = package.get("needs_scaling", False)

scaler_path = ROOT / "dataset" / "processed" / "feature_scaler.pkl"

scaler = None

if needs_scaling:
    if not scaler_path.exists():
        raise FileNotFoundError("Feature scaler not found")
    scaler = joblib.load(scaler_path)

print("Model:", package["model_name"])

# 4. Load historical electricity data
print("\n2. Loading historical data...")

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

df = df.sort_values("datetime")
df = df.drop_duplicates(subset="datetime")
df = df.set_index("datetime")

target = "consumption_kwh"

if target not in df.columns:
    raise ValueError(f"Target column {target} not found")

history = df[target].dropna().copy()

if len(history) < 721:
    raise ValueError("At least 721 hourly records are required")

print("Historical records:", len(history))
print("Last recorded time:", history.index[-1])

# 5. Function to create future features
def create_features(timestamp, values):

    previous = values.values

    row = {
        "lag_1": previous[-1],
        "lag_24": previous[-24],
        "lag_48": previous[-48],
        "lag_168": previous[-168],

        "rolling_mean_24": np.mean(previous[-24:]),
        "rolling_mean_168": np.mean(previous[-168:]),
        "rolling_mean_720": np.mean(previous[-720:]),

        "rolling_std_24": np.std(previous[-24:], ddof=1),

        "hour": timestamp.hour,
        "day": timestamp.day,
        "day_of_week": timestamp.dayofweek,
        "month": timestamp.month,

        "weekend": int(timestamp.dayofweek >= 5),

        "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
        "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),

        "month_sin": np.sin(2 * np.pi * timestamp.month / 12),
        "month_cos": np.cos(2 * np.pi * timestamp.month / 12)
    }

    return pd.DataFrame([row], columns=features)

# 6. Generate hourly predictions
print("\n3. Predicting next 7 days...")

future_predictions = []

future_time = history.index[-1] + pd.Timedelta(hours=1)

for i in range(24 * 7):

    X_future = create_features(future_time, history)

    if needs_scaling:
        X_input = scaler.transform(X_future)
    else:
        X_input = X_future

    prediction = float(model.predict(X_input)[0])

    # Electricity consumption cannot be negative
    prediction = max(0, prediction)

    future_predictions.append({
        "datetime": future_time,
        "predicted_kwh": prediction
    })

    # Add prediction to history for recursive forecasting
    history.loc[future_time] = prediction

    future_time += pd.Timedelta(hours=1)

# 7. Convert hourly predictions into daily predictions
print("\n4. Calculating daily consumption...")

hourly_df = pd.DataFrame(future_predictions)

hourly_df["date"] = hourly_df["datetime"].dt.date

daily_df = (
    hourly_df.groupby("date")["predicted_kwh"]
    .sum()
    .reset_index()
)

daily_df.columns = [
    "Date",
    "Predicted_Consumption_kWh"
]

daily_df["Predicted_Consumption_kWh"] = (
    daily_df["Predicted_Consumption_kWh"].round(3)
)

# 8. Display predictions
print("\n======================================")
print(" NEXT 7 DAYS ELECTRICITY PREDICTION")
print("======================================")

print(daily_df.to_string(index=False))

# 9. Save predictions
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

daily_df.to_csv(OUTPUT_PATH, index=False)

hourly_df.to_csv(
    ROOT / "prediction" / "hourly_predictions.csv",
    index=False
)

print("\n======================================")
print(" PREDICTION COMPLETED SUCCESSFULLY")
print("======================================")

print("Daily predictions saved:", OUTPUT_PATH)
print("Hourly predictions saved successfully")
