
import pandas as pd
import numpy as np
from pathlib import Path

print("Starting Advanced Feature Engineering...")

# Project paths
project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root / "dataset" / "processed"
    / "electricity_hourly_cleaned.csv"
)

output_file = (
    project_root / "dataset" / "processed"
    / "electricity_features.csv"
)

# Load cleaned dataset
df = pd.read_csv(input_file, parse_dates=["datetime"])

df = df.sort_values("datetime")
df = df.set_index("datetime")

# Target variable
target = "consumption_kwh"

print("\n1. Creating lag features...")

# Previous consumption
for lag in [1, 24, 48, 168]:
    df[f"lag_{lag}"] = df[target].shift(lag)

print("\n2. Creating rolling averages...")

# Shift first to avoid using the current target
previous = df[target].shift(1)

df["rolling_mean_24"] = previous.rolling(24).mean()
df["rolling_mean_168"] = previous.rolling(168).mean()
df["rolling_mean_720"] = previous.rolling(720).mean()

# Rolling standard deviation
df["rolling_std_24"] = previous.rolling(24).std()

print("\n3. Creating time features...")

# Calendar features
df["hour"] = df.index.hour
df["day"] = df.index.day
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["weekend"] = (df["day_of_week"] >= 5).astype(int)

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

print("\n4. Removing rows with missing engineered features...")

df = df.dropna()

# Save features
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file)

print("\nFeature Engineering Completed!")
print("Final shape:", df.shape)

print("\nCreated Features:")
print(df.columns.tolist())

print("\nSaved to:", output_file)
