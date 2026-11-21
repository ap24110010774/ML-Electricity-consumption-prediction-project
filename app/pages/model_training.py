
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def show_model_training_page():

    st.title("🤖 Electricity Consumption Model Training")

    df = st.session_state.get("processed_dataset")

    if df is None:
        st.warning("Please preprocess your dataset first.")
        return

    date_col = st.session_state.get("date_column")
    target_col = st.session_state.get("target_column")
    frequency = st.session_state.get("dataset_frequency")

    if date_col not in df.columns or target_col not in df.columns:
        st.error("Invalid date or target column.")
        return

    if frequency not in ["Hourly", "Daily", "Monthly"]:
        st.error("Please select a valid dataset frequency.")
        return

    data = df.copy()

    data[date_col] = pd.to_datetime(
        data[date_col], errors="coerce"
    )

    data[target_col] = pd.to_numeric(
        data[target_col], errors="coerce"
    )

    data = data.dropna(
        subset=[date_col, target_col]
    ).sort_values(date_col)

    # Configure lag features based on frequency
    if frequency == "Hourly":
        lag_periods = [1, 2, 24, 48, 168]
        rolling_periods = [24, 168]

    elif frequency == "Daily":
        lag_periods = [1, 2, 3, 7, 14]
        rolling_periods = [7, 14]

    else:
        lag_periods = [1, 2, 3, 6]
        rolling_periods = [3, 6]

    if len(data) < max(lag_periods) + 10:
        st.error(
            f"Not enough records for {frequency.lower()} training. "
            f"Please provide at least {max(lag_periods) + 10} records."
        )
        return

    # Calendar features
    dates = data[date_col]

    data["year"] = dates.dt.year
    data["month"] = dates.dt.month
    data["day"] = dates.dt.day
    data["dayofweek"] = dates.dt.dayofweek
    data["dayofyear"] = dates.dt.dayofyear
    data["hour"] = dates.dt.hour
    data["quarter"] = dates.dt.quarter

    feature_cols = [
        "year", "month", "day",
        "dayofweek", "dayofyear",
        "hour", "quarter"
    ]

    # Lag features
    for lag in lag_periods:
        column = f"lag_{lag}"
        data[column] = data[target_col].shift(lag)
        feature_cols.append(column)

    # Rolling features using only previous observations
    for window in rolling_periods:
        column = f"rolling_mean_{window}"
        data[column] = (
            data[target_col]
            .shift(1)
            .rolling(window=window)
            .mean()
        )
        feature_cols.append(column)

    data = data.dropna(subset=feature_cols + [target_col])

    if len(data) < 20:
        st.error("Not enough usable records after feature creation.")
        return

    X = data[feature_cols]
    y = data[target_col]

    # Chronological split
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Frequency", frequency)
    col2.metric("Training Records", len(X_train))
    col3.metric("Testing Records", len(X_test))

    st.write("Target column:", target_col)

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42
        )
    }

    if st.button("Train Machine Learning Models",
                 use_container_width=True):

        results = []
        trained_models = {}

        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)

            rmse = np.sqrt(
                mean_squared_error(y_test, predictions)
            )

            r2 = r2_score(y_test, predictions)

            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2 Score": r2
            })

            trained_models[name] = model

            progress.progress((i + 1) / len(models))

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            "RMSE", ascending=True
        )

        best_name = results_df.iloc[0]["Model"]
        best_model = trained_models[best_name]

        # Save model in project root/model
        project_root = Path(__file__).resolve().parents[2]

        model_folder = project_root / "model"
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = model_folder / "best_electricity_model.pkl"

        model_package = {
            "model": best_model,
            "features": feature_cols,
            "date_column": date_col,
            "target_column": target_col,
            "frequency": frequency,
            "lag_periods": lag_periods,
            "rolling_periods": rolling_periods,
            "last_date": data[date_col].max(),
            "last_values": data[target_col].tail(
                max(max(lag_periods), max(rolling_periods))
            ).tolist()
        }

        joblib.dump(model_package, model_path)

        results_df.to_csv(
            model_folder / "model_comparison.csv",
            index=False
        )

        st.session_state["model_trained"] = True
        st.session_state["best_model_name"] = best_name
        st.session_state["model_results"] = results_df

        st.success("Model training completed!")

        st.subheader("Best Model")
        st.success(f"Best performing model: {best_name}")

        st.write(f"Saved model: {model_path}")

        st.subheader("Model Performance Comparison")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("RMSE Comparison")
        st.bar_chart(
            results_df.set_index("Model")[["RMSE"]]
        )

        st.download_button(
            "Download Model Comparison",
            data=results_df.to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv"
        )
