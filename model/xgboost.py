# Electricity Consumption Prediction - Model Training and Comparison

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load preprocessed dataset
df = pd.read_csv(
    "../electricity_preprocessed_hourly.csv",
    parse_dates=["datetime"],
    index_col="datetime"
)

TARGET = "target_consumption_kwh"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 2. Chronological 80:20 split (do not shuffle time-series data)
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 3. Define models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )
}

# 4. Train, predict and evaluate
results = []
trained_models = {}

for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

# 5. Compare models (lower RMSE is better)
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison_results.csv", index=False)

# 6. Save the best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "features": list(X.columns),
        "target": TARGET
    },
    "best_electricity_model.pkl"
)

print("\\nBest model:", best_model_name)
print("Saved as best_electricity_model.pkl")

# 7. Save actual vs predicted values for inspection