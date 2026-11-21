
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Starting Model Evaluation...")

# Project paths
ROOT = Path(__file__).resolve().parent.parent

data_path = (
    ROOT / "dataset" / "processed"
    / "prepared_data.pkl"
)

model_path = (
    ROOT / "model" / "best_tuned_model.pkl"
)

# Load data and model
data = joblib.load(data_path)
package = joblib.load(model_path)

model = package["model"]
needs_scaling = package.get("needs_scaling", False)

X_test = data["X_test"]
y_test = data["y_test"]

# Apply scaling only if required
if needs_scaling:
    scaler = joblib.load(
        ROOT / "dataset" / "processed"
        / "feature_scaler.pkl"
    )
    X_test_input = scaler.transform(X_test)
else:
    X_test_input = X_test

# Generate predictions
print("\n1. Generating predictions...")

y_pred = model.predict(X_test_input)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("---------------------------")
print("Model:", package["model_name"])
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))
print("R2 Percentage:", round(r2 * 100, 2), "%")

# Create comparison dataframe
comparison = pd.DataFrame({
    "Actual": np.asarray(y_test),
    "Predicted": y_pred
})

comparison["Error"] = (
    comparison["Actual"] - comparison["Predicted"]
)

comparison.to_csv(
    ROOT / "model" / "actual_vs_predicted.csv",
    index=False
)

# Actual vs predicted graph
plt.figure(figsize=(10, 5))

plt.plot(
    comparison["Actual"].values[:200],
    label="Actual",
    linewidth=2
)

plt.plot(
    comparison["Predicted"].values[:200],
    label="Predicted",
    linewidth=2
)

plt.title("Actual vs Predicted Electricity Consumption")
plt.xlabel("Test Records")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "model" / "actual_vs_predicted.png"
)

plt.show()

# Residual error graph
plt.figure(figsize=(10, 5))

plt.scatter(
    comparison["Predicted"],
    comparison["Error"],
    alpha=0.5
)

plt.axhline(y=0, linestyle="--")

plt.title("Prediction Error Analysis")
plt.xlabel("Predicted Consumption (kWh)")
plt.ylabel("Prediction Error (kWh)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "model" / "prediction_errors.png"
)

plt.show()

print("\nModel evaluation completed!")
print("Evaluation files saved successfully.")
