
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    ROOT / "prediction"
    / "daily_electricity_predictions.csv"
)

COMPARISON_FILE = (
    ROOT / "model" / "model_comparison.csv"
)

EVALUATION_FILE = (
    ROOT / "model" / "actual_vs_predicted.csv"
)

# Page configuration
st.set_page_config(
    page_title="Electricity Prediction",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ Electricity Consumption Prediction")
st.write(
    "Machine Learning Based Daily Electricity Forecasting System"
)

st.divider()

# Load prediction data
if not PREDICTION_FILE.exists():
    st.error(
        "Prediction file not found. "
        "Run daily_prediction.py first."
    )
    st.stop()

predictions = pd.read_csv(PREDICTION_FILE)

predictions["Date"] = pd.to_datetime(
    predictions["Date"]
)

# Sidebar
st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Daily Predictions",
        "Model Performance",
        "Prediction Analysis"
    ]
)

# Overview
if page == "Overview":

    st.subheader("Electricity Forecast Overview")

    values = predictions[
        "Predicted_Consumption_kWh"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Daily Consumption",
        f"{values.mean():.2f} kWh"
    )

    col2.metric(
        "Highest Daily Consumption",
        f"{values.max():.2f} kWh"
    )

    col3.metric(
        "Lowest Daily Consumption",
        f"{values.min():.2f} kWh"
    )

    st.divider()

    fig = px.line(
        predictions,
        x="Date",
        y="Predicted_Consumption_kWh",
        markers=True,
        title="Daily Electricity Consumption Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Summary")

    st.dataframe(
        predictions,
        use_container_width=True
    )

# Daily predictions
elif page == "Daily Predictions":

    st.subheader("Daily Electricity Predictions")

    fig = px.bar(
        predictions,
        x="Date",
        y="Predicted_Consumption_kWh",
        title="Daily Predicted Electricity Consumption"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        predictions,
        use_container_width=True
    )

    csv = predictions.to_csv(index=False)

    st.download_button(
        "Download Daily Predictions",
        data=csv,
        file_name="daily_predictions.csv",
        mime="text/csv"
    )

# Model performance
elif page == "Model Performance":

    st.subheader("Machine Learning Model Comparison")

    if COMPARISON_FILE.exists():

        comparison = pd.read_csv(COMPARISON_FILE)

        st.dataframe(
            comparison,
            use_container_width=True
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="R2 Score",
            title="Model R2 Score Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning("Model comparison file not found.")

# Prediction analysis
elif page == "Prediction Analysis":

    st.subheader("Actual vs Predicted Consumption")

    if EVALUATION_FILE.exists():

        evaluation = pd.read_csv(EVALUATION_FILE)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=evaluation["Actual"].head(200),
                mode="lines",
                name="Actual"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=evaluation["Predicted"].head(200),
                mode="lines",
                name="Predicted"
            )
        )

        fig.update_layout(
            title="Actual vs Predicted Electricity Consumption",
            xaxis_title="Test Records",
            yaxis_title="Consumption (kWh)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning(
            "Run evaluate_model.py to generate evaluation data."
        )

st.divider()

st.caption(
    "Electricity Consumption Prediction | Machine Learning Project"
)
