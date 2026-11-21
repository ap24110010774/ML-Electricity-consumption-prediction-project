
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Starting Electricity Consumption Model Training...")

# Project paths
project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root / "dataset" / "processed"
    / "prepared_data.pkl"
)

model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)

# Load prepared data
data = joblib.load(data_path)

X_train = data["X_train"]
X_test = data["X_test"]

X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]

y_train = data["y_train"]
y_test = data["y_test"]

# Define models
models = {
    "Linear Regression": (
        LinearRegression(), True
    ),

    "Ridge Regression": (
        Ridge(alpha=1.0), True
    ),

    "Decision Tree": (
        DecisionTreeRegressor(
            max_depth=15,
            random_state=42
        ), False
    ),

    "Random Forest": (
        RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ), False
    ),

    "Extra Trees": (
        ExtraTreesRegressor(
            n_estimators=150,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ), False
    ),

    "Hist Gradient Boosting": (
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42
        ), False
    )
}

results = []
trained_models = {}

# Train and evaluate
for name, (model, needs_scaling) in models.items():

    print(f"\nTraining {name}...")

    if needs_scaling:
        Xtr = X_train_scaled
        Xte = X_test_scaled
    else:
        Xtr = X_train
        Xte = X_test

    model.fit(Xtr, y_train)

    predictions = model.predict(Xte)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test, predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

# Compare results
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="R2 Score",
    ascending=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

# Select best model
best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

# Save best model and its scaling requirement
best_package = {
    "model": best_model,
    "model_name": best_name,
    "needs_scaling": models[best_name][1],
    "features": data["features"]
}

joblib.dump(
    best_package,
    model_folder / "best_electricity_model.pkl"
)

# Save comparison report
comparison.to_csv(
    model_folder / "model_comparison.csv",
    index=False
)

print("\nBEST MODEL:", best_name)
print("Saved best model successfully!")
print("Saved comparison report!")
