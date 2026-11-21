
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("Starting Prediction Visualization...")

# Project paths
ROOT = Path(__file__).resolve().parent.parent

file_path = (
    ROOT / "prediction"
    / "daily_electricity_predictions.csv"
)

# Check file
if not file_path.exists():
    raise FileNotFoundError(
        "Prediction CSV not found. Run daily_prediction.py first."
    )

# Load predictions
df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

dates = df["Date"]
consumption = df["Predicted_Consumption_kWh"]

# Statistics
average = consumption.mean()
maximum = consumption.max()
minimum = consumption.min()

print("\nDAILY PREDICTION SUMMARY")
print("Average:", round(average, 3), "kWh")
print("Maximum:", round(maximum, 3), "kWh")
print("Minimum:", round(minimum, 3), "kWh")

# 1. Daily consumption bar chart
plt.figure(figsize=(11, 5))

plt.bar(
    dates.dt.strftime("%d-%b"),
    consumption
)

plt.title("Daily Electricity Consumption Prediction")
plt.xlabel("Date")
plt.ylabel("Predicted Consumption (kWh)")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "prediction" / "daily_bar_chart.png"
)

plt.show()

# 2. Daily consumption line chart
plt.figure(figsize=(11, 5))

plt.plot(
    dates,
    consumption,
    marker="o",
    linewidth=2
)

plt.title("7-Day Electricity Consumption Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Consumption (kWh)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "prediction" / "daily_line_chart.png"
)

plt.show()

print("\nVisualization completed successfully!")
print("Graphs saved in prediction folder.")
