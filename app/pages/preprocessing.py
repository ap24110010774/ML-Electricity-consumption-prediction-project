
import streamlit as st
import pandas as pd


def show_preprocessing_page():

    st.title("⚙️ Electricity Data Preprocessing")

    df = st.session_state.get("uploaded_dataset")

    if df is None:
        st.warning("Please upload your dataset first.")
        return

    st.subheader("Original Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    date_col = st.selectbox(
        "Select Date/Time Column",
        df.columns
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_columns:
        st.error("No numeric columns found.")
        return

    target_col = st.selectbox(
        "Select Electricity Consumption Column",
        numeric_columns
    )

    st.subheader("Dataset Settings")

    input_frequency = st.selectbox(
        "Original Dataset Frequency",
        ["Hourly", "Daily", "Monthly"]
    )

    output_frequency = st.selectbox(
        "Required Forecast Frequency",
        ["Hourly", "Daily", "Monthly"]
    )

    value_type = st.selectbox(
        "What does your target column represent?",
        [
            "Energy per interval (kWh)",
            "Average power (kW)"
        ]
    )

    missing_method = st.selectbox(
        "Missing Value Method",
        [
            "Remove missing rows",
            "Fill with mean",
            "Fill with median",
            "Forward fill",
            "Linear interpolation"
        ]
    )

    if st.button("Preprocess Dataset", use_container_width=True):

        data = df.copy()

        data[date_col] = pd.to_datetime(
            data[date_col],
            errors="coerce"
        )

        data[target_col] = pd.to_numeric(
            data[target_col],
            errors="coerce"
        )

        data = data.dropna(subset=[date_col])
        data = data.sort_values(date_col)

        # Handle missing values
        if missing_method == "Remove missing rows":
            data = data.dropna(subset=[target_col])

        elif missing_method == "Fill with mean":
            data[target_col] = data[target_col].fillna(
                data[target_col].mean()
            )

        elif missing_method == "Fill with median":
            data[target_col] = data[target_col].fillna(
                data[target_col].median()
            )

        elif missing_method == "Forward fill":
            data[target_col] = data[target_col].ffill()

        elif missing_method == "Linear interpolation":
            data[target_col] = data[target_col].interpolate()

        data = data.dropna(subset=[target_col])

        if data.empty:
            st.error("No valid data available after preprocessing.")
            return

        # Check frequency compatibility
        if (
            input_frequency == "Daily"
            and output_frequency == "Hourly"
        ) or (
            input_frequency == "Monthly"
            and output_frequency != "Monthly"
        ):
            st.error(
                "This conversion is not supported reliably. "
                "A lower-frequency dataset cannot provide "
                "genuine higher-frequency observations."
            )
            return

        # Prepare time series
        series = data.set_index(date_col)[target_col]

        # Convert power readings to energy estimates
        if value_type == "Average power (kW)":

            if input_frequency == "Hourly":
                # Each hourly average represents approximately 1 hour
                series = series * 1

            elif input_frequency == "Daily":
                # Daily average power converted to estimated daily kWh
                series = series * 24

            elif input_frequency == "Monthly":
                # Monthly average power converted using calendar hours
                hours = series.index.days_in_month * 24
                series = series * hours

        # Aggregate to required frequency
        if input_frequency == "Hourly":

            if output_frequency == "Hourly":
                processed = series.resample("h").sum(min_count=1)

            elif output_frequency == "Daily":
                processed = series.resample("D").sum(min_count=1)

            else:
                processed = series.resample("MS").sum(min_count=1)

        elif input_frequency == "Daily":

            if output_frequency == "Daily":
                processed = series.resample("D").sum(min_count=1)

            else:
                processed = series.resample("MS").sum(min_count=1)

        else:
            processed = series.resample("MS").sum(min_count=1)

        processed = processed.dropna()

        if len(processed) < 10:
            st.error(
                "Not enough observations after aggregation. "
                "Please check your dataset and selected frequency."
            )
            return

        result = processed.rename(
            target_col
        ).reset_index()

        result.columns = [date_col, target_col]

        # Save processed data and settings
        st.session_state["processed_dataset"] = result
        st.session_state["date_column"] = date_col
        st.session_state["target_column"] = target_col
        st.session_state["dataset_frequency"] = output_frequency
        st.session_state["value_type"] = value_type

        st.success("Preprocessing and aggregation completed!")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(result))
        col2.metric("Original Frequency", input_frequency)
        col3.metric("Output Frequency", output_frequency)

        st.subheader("Processed Dataset")
        st.dataframe(result.head(20), use_container_width=True)

        st.subheader("Consumption Statistics")
        st.write(result[target_col].describe())

        st.subheader("Consumption Trend")
        st.line_chart(
            result.set_index(date_col)[target_col]
        )

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Processed Dataset",
            data=csv,
            file_name="electricity_preprocessed.csv",
            mime="text/csv"
        )
