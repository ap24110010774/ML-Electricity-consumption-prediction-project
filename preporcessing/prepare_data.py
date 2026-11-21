
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

print("Starting Feature Selection and Scaling...")

# Project paths
project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root / "dataset" / "processed"
    / "electricity_features.csv"
)

output_folder = project_root / "dataset" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

# Load feature dataset
df = pd.read_csv(input_file, parse_dates=["datetime"])

df = df.sort_values("datetime")

print("\n1. Dataset loaded:", df.shape)

# Select useful features
features = [
    "lag_1",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "rolling_mean_720",
    "rolling_std_24",
    "hour",
    "day",
    "day_of_week",
    "month",
    "weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]

target = "consumption_kwh"

# Check required columns
missing = [
    col for col in features + [target]
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Remove missing and infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

print("\n2. Selected features:", len(features))
print(features)

# Chronological train-test split
print("\n3. Splitting dataset...")

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Feature scaling
print("\n4. Applying StandardScaler...")

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using training scaler
X_test_scaled = scaler.transform(X_test)

# Save prepared data
prepared_data = {
    "X_train": X_train,
    "X_test": X_test,
    "X_train_scaled": X_train_scaled,
    "X_test_scaled": X_test_scaled,
    "y_train": y_train,
    "y_test": y_test,
    "features": features
}

joblib.dump(
    prepared_data,
    output_folder / "prepared_data.pkl"
)

joblib.dump(
    scaler,
    output_folder / "feature_scaler.pkl"
)

print("\n5. Feature scaling completed!")

print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

print("\nSaved files:")
print("prepared_data.pkl")
print("feature_scaler.pkl")

print("\nFeature preparation completed successfully!")
