
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Starting Hyperparameter Tuning...")

# Project paths
root = Path(__file__).resolve().parent.parent

data = joblib.load(
    root / "dataset" / "processed" / "prepared_data.pkl"
)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Time-series cross validation
cv = TimeSeriesSplit(n_splits=3)

# Models and parameter combinations
searches = {
    "Hist Gradient Boosting": (
        HistGradientBoostingRegressor(random_state=42),
        {
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "max_iter": [100, 200, 300],
            "max_leaf_nodes": [15, 20, 31, 40],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0]
        }
    ),

    "Extra Trees": (
        ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1
        ),
        {
            "n_estimators": [100, 150, 200],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    )
}

results = []

best_score = float("-inf")
best_model = None
best_name = None

for name, (model, params) in searches.items():

    print(f"\nTuning {name}...")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=12,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_

    predictions = tuned_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Best parameters:", search.best_params_)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    if r2 > best_score:
        best_score = r2
        best_model = tuned_model
        best_name = name

# Save results
result_df = pd.DataFrame(results)
result_df.to_csv(
    root / "model" / "tuning_comparison.csv",
    index=False
)

# Save best tuned model
joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "needs_scaling": False,
        "features": data["features"]
    },
    root / "model" / "best_tuned_model.pkl"
)

print("\nTUNING COMPLETED")
print("Best model:", best_name)
print("Best R2 Score:", best_score)
print("Saved: best_tuned_model.pkl")
