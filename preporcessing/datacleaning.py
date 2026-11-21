
import pandas as pd
import numpy as np
from pathlib import Path

print("Starting Electricity Data Preprocessing...")

# Project directory
project_root = Path(__file__).resolve().parent.parent

# Input and output paths
input_file = project_root / "dataset" / "raw" / "household_power_consumption.txt"
output_file = project_root / "dataset" / "processed" / "electricity_hourly_cleaned.csv"

# Check dataset
if not input_file.exists():
    raise FileNotFoundError(f"Dataset not found: {input_file}")

# Load original dataset
print("\n1. Loading dataset...")

df = pd.read_csv(
    input_file,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("Original shape:", df.shape)
print(df.head())

# Combine Date and Time
print("\n2. Converting date and time...")

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

df.drop(columns=["Date", "Time"], inplace=True)

# Convert columns to numeric
print("\n3. Converting columns to numeric...")

for col in df.columns:
    if col != "datetime":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid timestamps
df.dropna(subset=["datetime"], inplace=True)

# Sort records
df = df.sort_values("datetime")

# Remove duplicate timestamps
df = df.drop_duplicates(subset=["datetime"])

# Missing value analysis
print("\n4. Checking missing values...")
print(df.isnull().sum())

# Remove rows where target is missing
df = df.dropna(subset=["Global_active_power"])

# Fill missing values in other numeric features
numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns] = df[numeric_columns].interpolate(
    method="linear",
    limit=5,
    limit_direction="forward"
)

# Remove remaining rows with missing values
df = df.dropna()

# Hourly aggregation
print("\n5. Converting minute readings into hourly data...")

df = df.set_index("datetime")

hourly = df.resample("h").agg({
    "Global_active_power": "sum",
    "Global_reactive_power": "mean",
    "Voltage": "mean",
    "Global_intensity": "mean",
    "Sub_metering_1": "sum",
    "Sub_metering_2": "sum",
    "Sub_metering_3": "sum"
})

# Convert summed minute power readings (kW) into hourly energy (kWh)
hourly["consumption_kwh"] = (
    hourly["Global_active_power"] / 60
)

hourly = hourly.dropna(subset=["consumption_kwh"])

# Save cleaned data
output_file.parent.mkdir(parents=True, exist_ok=True)

hourly.to_csv(output_file)

print("\n6. Preprocessing completed!")

print("Final dataset shape:", hourly.shape)
print("Missing values:")
print(hourly.isnull().sum())

print("\nFinal dataset preview:")
print(hourly.head())

print("\nSaved file:", output_file)
print("\nData preprocessing completed successfully!")
