
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


def show_predictions_page():

    st.title("⚡ Electricity Consumption Forecasting")

    st.write(
        "Generate future electricity consumption forecasts "
        "using your trained machine learning model."
    )

    # Locate saved model
    project_root = Path(__file__).resolve().parents[2]
    model_path = (
        project_root / "model" / "best_electricity_model.pkl"
    )

    if not model_path.exists():
        st.error("Model not found. Please train your model first.")
        return

    try:
        package = joblib.load(model_path)
        model = package["model"]
        feature_cols = package["features"]
        frequency = package["frequency"]
        target_col = package["target_column"]
        last_date = pd.Timestamp(package["last_date"])
        history = list(package["last_values"])

        lag_periods = package["lag_periods"]
        rolling_periods = package["rolling_periods"]

    except Exception as e:
        st.error(f"Unable to load model: {e}")
        return

    # Verify model information
    if not history:
        st.error("Training history is missing. Retrain the model.")
        return

    st.subheader("Trained Model Information")

    col1, col2 = st.columns(2)

    col1.metric("Dataset Frequency", frequency)
    col2.metric("Target Variable", target_col)

    st.subheader("Forecast Settings")

    if frequency == "Hourly":
        forecast_periods = st.number_input(
            "Number of hours to forecast",
            min_value=1,
            max_value=720,
            value=24
        )

        interval = "h"

    elif frequency == "Daily":
        forecast_periods = st.number_input(
            "Number of days to forecast",
            min_value=1,
            max_value=365,
            value=30
        )

        interval = "D"

    else:
        forecast_periods = st.number_input(
            "Number of months to forecast",
            min_value=1,
            max_value=36,
            value=12
        )

        interval = "MS"

    if st.button(
        "Generate Electricity Forecast",
        use_container_width=True
    ):

        predictions = []
        forecast_dates = []

        values = list(history)

        current_date = last_date

        progress = st.progress(0)

        for i in range(int(forecast_periods)):

            # Generate next timestamp
            if frequency == "Hourly":
                next_date = current_date + pd.Timedelta(hours=1)

            elif frequency == "Daily":
                next_date = current_date + pd.Timedelta(days=1)

            else:
                next_date = current_date + pd.offsets.MonthBegin(1)

            # Calendar features
            features = {
                "year": next_date.year,
                "month": next_date.month,
                "day": next_date.day,
                "dayofweek": next_date.dayofweek,
                "dayofyear": next_date.dayofyear,
                "hour": next_date.hour,
                "quarter": next_date.quarter
            }

            # Lag features
            for lag in lag_periods:

                if len(values) < lag:
                    st.error(
                        "Insufficient historical values for forecasting."
                    )
                    return

                features[f"lag_{lag}"] = values[-lag]

            # Rolling mean features
            for window in rolling_periods:

                if len(values) < window:
                    st.error(
                        "Insufficient history for rolling features."
                    )
                    return

                features[f"rolling_mean_{window}"] = float(
                    np.mean(values[-window:])
                )

            # Arrange features in training order
            input_data = pd.DataFrame(
                [[features[col] for col in feature_cols]],
                columns=feature_cols
            )

            prediction = float(
                model.predict(input_data)[0]
            )

            prediction = max(0, prediction)

            predictions.append(prediction)
            forecast_dates.append(next_date)

            # Use predicted value for next forecast
            values.append(prediction)

            current_date = next_date

            progress.progress(
                (i + 1) / int(forecast_periods)
            )

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Consumption": predictions
        })

        st.session_state["forecast_results"] = forecast_df

        st.success("Forecast generated successfully!")

        st.subheader("Forecast Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Forecast Records",
            len(forecast_df)
        )

        col2.metric(
            "Total Forecast Consumption",
            f"{forecast_df['Predicted Consumption'].sum():.2f} kWh"
        )

        # Display chart
        st.subheader("Electricity Consumption Forecast")

        st.line_chart(
            forecast_df.set_index("Date")[
                "Predicted Consumption"
            ]
        )

        # Monthly aggregation for hourly/daily forecasts
        if frequency in ["Hourly", "Daily"]:

            monthly_df = forecast_df.copy()

            monthly_df["Month"] = (
                monthly_df["Date"].dt.to_period("M")
            )

            monthly_summary = (
                monthly_df.groupby("Month")[
                    "Predicted Consumption"
                ].sum().reset_index()
            )

            monthly_summary["Month"] = (
                monthly_summary["Month"].astype(str)
            )

            st.subheader("Monthly Consumption Summary")

            st.dataframe(
                monthly_summary,
                use_container_width=True
            )

        else:

            st.subheader("Monthly Consumption Forecast")

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

        # Download forecast
        csv = forecast_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name="electricity_forecast.csv",
            mime="text/csv"
        )
