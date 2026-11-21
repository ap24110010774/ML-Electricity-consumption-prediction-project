
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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Electricity Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. COLUMN DETECTION
# ==========================================

def detect_columns(df):

    date_col = None
    target_col = None

    date_names = [
        "datetime", "date_time",
        "timestamp", "date", "time"
    ]

    target_names = [
        "global_active_power",
        "consumption",
        "electricity_consumption",
        "energy", "power", "usage", "kwh"
    ]

    for col in df.columns:

        if str(col).strip().lower() in date_names:
            date_col = col
            break

    if date_col is None:

        for col in df.columns:

            parsed = pd.to_datetime(
                df[col], errors="coerce"
            )

            if parsed.notna().mean() > 0.8:
                date_col = col
                break

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    for col in numeric_cols:

        name = str(col).strip().lower()

        if any(
            target in name
            for target in target_names
        ):
            target_col = col
            break

    if target_col is None and numeric_cols:
        target_col = numeric_cols[0]

    return date_col, target_col


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

def prepare_hourly_data(df, date_col, target_col):

    data = df.copy()

    if (
        str(date_col).strip().lower() == "date"
        and "Time" in data.columns
    ):

        combined = (
            data[date_col].astype(str)
            + " "
            + data["Time"].astype(str)
        )

        data[date_col] = pd.to_datetime(
            combined, errors="coerce"
        )

    else:

        data[date_col] = pd.to_datetime(
            data[date_col], errors="coerce"
        )

    data[target_col] = pd.to_numeric(
        data[target_col], errors="coerce"
    )

    data = data.dropna(
        subset=[date_col, target_col]
    )

    data = data.sort_values(date_col)

    if data.empty:
        raise ValueError("No valid records found.")

    series = (
        data.groupby(date_col)[target_col]
        .mean()
        .sort_index()
    )

    differences = (
        series.index.to_series().diff().dropna()
    )

    if differences.empty:
        raise ValueError("Not enough timestamp records.")

    if differences.median() > pd.Timedelta(hours=1, minutes=30):

        raise ValueError(
            "Hourly forecasting requires hourly observations."
        )

    series = series.resample("h").mean()

    series = series.interpolate(
        method="time",
        limit=3
    ).dropna()

    if len(series) < 200:
        raise ValueError(
            "At least 200 valid hourly observations are recommended."
        )

    series.name = target_col

    return series


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

def create_features(series):

    data = pd.DataFrame({
        "target": series
    })

    dates = data.index

    data["hour"] = dates.hour
    data["day"] = dates.day
    data["month"] = dates.month
    data["dayofweek"] = dates.dayofweek
    data["dayofyear"] = dates.dayofyear

    data["is_weekend"] = (
        dates.dayofweek >= 5
    ).astype(int)

    # Cyclical features
    data["hour_sin"] = np.sin(
        2 * np.pi * dates.hour / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * dates.hour / 24
    )

    data["day_sin"] = np.sin(
        2 * np.pi * dates.dayofweek / 7
    )

    data["day_cos"] = np.cos(
        2 * np.pi * dates.dayofweek / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * dates.month / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * dates.month / 12
    )

    # Lag features
    for lag in [1, 2, 3, 24, 48, 168]:

        data[f"lag_{lag}"] = (
            data["target"].shift(lag)
        )

    # Rolling features
    for window in [6, 24, 48, 168]:

        data[f"rolling_mean_{window}"] = (
            data["target"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    data["rolling_std_24"] = (
        data["target"]
        .shift(1)
        .rolling(24)
        .std()
    )

    return data.dropna()


# ==========================================
# 5. MODEL TRAINING AND EVALUATION
# ==========================================

def train_best_model(series):

    data = create_features(series)

    feature_cols = [
        col for col in data.columns
        if col != "target"
    ]

    X = data[feature_cols]
    y = data["target"]

    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    progress = st.progress(0)

    for i, (name, model) in enumerate(models.items()):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({

            "Model": name,

            "MAE": mean_absolute_error(
                y_test, predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test, predictions
                )
            ),

            "R2 Score": r2_score(
                y_test, predictions
            )

        })

        trained_models[name] = model

        progress.progress(
            (i + 1) / len(models)
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "RMSE", ascending=True
    )

    # Select best model
    best_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_name]

    # Generate test predictions using the selected model
    best_test_predictions = best_model.predict(X_test)

    evaluation_df = pd.DataFrame({
        "Actual Consumption": y_test.values,
        "Predicted Consumption": best_test_predictions
    }, index=X_test.index)

    # Save trained model
    package = {

        "model": best_model,
        "features": feature_cols,
        "last_values": series.tail(168).tolist(),
        "last_date": series.index[-1],
        "target_column": series.name,
        "frequency": "Hourly"

    }

    model_path = (
        MODEL_DIR / "best_electricity_model.pkl"
    )

    joblib.dump(package, model_path)

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    return package, results_df, best_name, evaluation_df


# ==========================================
# 6. FUTURE FORECASTING
# ==========================================

def generate_forecast(package, hours):

    model = package["model"]

    feature_cols = package["features"]

    values = list(package["last_values"])

    last_date = pd.Timestamp(
        package["last_date"]
    )

    dates = []
    predictions = []

    for _ in range(hours):

        next_date = (
            last_date + pd.Timedelta(hours=1)
        )

        features = {

            "hour": next_date.hour,
            "day": next_date.day,
            "month": next_date.month,
            "dayofweek": next_date.dayofweek,
            "dayofyear": next_date.dayofyear,

            "is_weekend": int(
                next_date.dayofweek >= 5
            ),

            "hour_sin": np.sin(
                2 * np.pi * next_date.hour / 24
            ),

            "hour_cos": np.cos(
                2 * np.pi * next_date.hour / 24
            ),

            "day_sin": np.sin(
                2 * np.pi * next_date.dayofweek / 7
            ),

            "day_cos": np.cos(
                2 * np.pi * next_date.dayofweek / 7
            ),

            "month_sin": np.sin(
                2 * np.pi * next_date.month / 12
            ),

            "month_cos": np.cos(
                2 * np.pi * next_date.month / 12
            )

        }

        for lag in [1, 2, 3, 24, 48, 168]:

            features[f"lag_{lag}"] = values[-lag]

        for window in [6, 24, 48, 168]:

            features[f"rolling_mean_{window}"] = (
                np.mean(values[-window:])
            )

        features["rolling_std_24"] = (
            np.std(values[-24:], ddof=1)
        )

        input_data = pd.DataFrame(
            [[features[col] for col in feature_cols]],
            columns=feature_cols
        )

        prediction = float(
            model.predict(input_data)[0]
        )

        prediction = max(0, prediction)

        values.append(prediction)
        predictions.append(prediction)
        dates.append(next_date)

        last_date = next_date

    return pd.DataFrame({

        "DateTime": dates,

        "Predicted Consumption": predictions

    })


# ==========================================
# 7. MAIN WEBSITE
# ==========================================

def main():

    st.title("⚡ Electricity Consumption Prediction")

    st.write(
        "Upload your electricity dataset and enter "
        "the number of hours you want to forecast."
    )

    uploaded_file = st.file_uploader(
        "Upload Electricity Dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is None:

        st.info("Upload a dataset to begin.")

        return

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

    except Exception as e:

        st.error(f"Unable to read dataset: {e}")

        return

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # Detect columns
    detected_date, detected_target = detect_columns(df)

    columns = list(df.columns)

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_cols:

        st.error("No numeric consumption column found.")

        return

    date_index = (
        columns.index(detected_date)
        if detected_date in columns else 0
    )

    target_index = (
        numeric_cols.index(detected_target)
        if detected_target in numeric_cols else 0
    )

    st.subheader("Confirm Dataset Columns")

    date_col = st.selectbox(
        "Date/Time Column",
        columns,
        index=date_index
    )

    target_col = st.selectbox(
        "Electricity Consumption Column",
        numeric_cols,
        index=target_index
    )

    hours = st.number_input(
        "How many hours do you want to predict?",
        min_value=1,
        max_value=720,
        value=24
    )

    # Run automatic workflow
    if st.button(
        "Automatically Train and Predict",
        use_container_width=True
    ):

        try:

            with st.spinner("Preparing dataset..."):

                series = prepare_hourly_data(
                    df, date_col, target_col
                )

            st.success(
                f"Prepared {len(series)} hourly records."
            )

            with st.spinner("Training ML models..."):

                package, results, best_name, evaluation_df = (
                    train_best_model(series)
                )

            st.success(
                f"Best model selected: {best_name}"
            )

            # Model performance
            st.subheader("Model Performance")

            st.dataframe(
                results,
                use_container_width=True
            )

            st.bar_chart(
                results.set_index("Model")[["RMSE"]]
            )

            # Actual vs Predicted graph
            st.subheader("Actual vs Predicted Consumption")

            st.write(
                "Comparison of actual test-set consumption "
                "with the selected model's predictions."
            )

            st.line_chart(
                evaluation_df,
                use_container_width=True
            )

            # Error analysis
            evaluation_df["Prediction Error"] = (
                evaluation_df["Actual Consumption"]
                - evaluation_df["Predicted Consumption"]
            )

            st.subheader("Prediction Error Analysis")

            col1, col2 = st.columns(2)

            col1.metric(
                "Average Absolute Error",
                f"{evaluation_df['Prediction Error'].abs().mean():.4f}"
            )

            col2.metric(
                "Maximum Absolute Error",
                f"{evaluation_df['Prediction Error'].abs().max():.4f}"
            )

            with st.expander("View actual and predicted values"):

                st.dataframe(
                    evaluation_df,
                    use_container_width=True
                )

            # Future forecast
            with st.spinner("Generating future forecast..."):

                forecast_df = generate_forecast(
                    package, int(hours)
                )

            st.success("Forecast generated successfully!")

            total = forecast_df[
                "Predicted Consumption"
            ].sum()

            st.subheader("Future Electricity Forecast")

            col1, col2, col3 = st.columns(3)

            col1.metric("Forecast Hours", int(hours))

            col2.metric("Total Forecast", f"{total:.2f}")

            col3.metric("Best Model", best_name)

            st.line_chart(
                forecast_df.set_index("DateTime")[
                    "Predicted Consumption"
                ]
            )

            st.subheader("Forecast Results")

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            # Monthly summary
            monthly = forecast_df.copy()

            monthly["Month"] = (
                monthly["DateTime"]
                .dt.to_period("M")
                .astype(str)
            )

            monthly_summary = (
                monthly.groupby("Month")[
                    "Predicted Consumption"
                ].sum().reset_index()
            )

            st.subheader("Monthly Consumption Summary")

            st.dataframe(
                monthly_summary,
                use_container_width=True
            )

            # Downloads
            st.download_button(
                "Download Forecast CSV",
                data=forecast_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="electricity_forecast.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download Model Comparison",
                data=results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv"
            )

            st.info(
                "Trained model saved at "
                "model/best_electricity_model.pkl"
            )

        except Exception as e:

            st.error(
                f"Processing or prediction failed: {e}"
            )


if __name__ == "__main__":

    main()
