# ⚡ Electricity Consumption Prediction Using Machine Learning

## 📌 Project Overview

Electricity Consumption Prediction is a Machine Learning-based application developed using Python and Streamlit.

The system allows users to upload electricity consumption datasets, automatically preprocess the data, train multiple Machine Learning models, compare their performance, and predict future electricity consumption based on the number of hours entered by the user.

The application also displays prediction graphs, model evaluation metrics, and downloadable forecast results.

## 🎯 Project Objectives

- Predict future electricity consumption using Machine Learning.
- Automatically preprocess uploaded electricity datasets.
- Train and compare multiple Machine Learning algorithms.
- Identify the best-performing model.
- Save the trained model in `.pkl` format.
- Generate hourly electricity consumption forecasts.
- Visualize actual and predicted consumption.
- Download prediction results as CSV files.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Website development |
| Pandas | Data processing |
| NumPy | Numerical calculations |
| Scikit-learn | Machine Learning algorithms and evaluation |
| Joblib | Saving and loading trained models |
| Plotly / Streamlit Charts | Data visualization |

## 🤖 Machine Learning Algorithms

The project compares the following algorithms:

1. Random Forest Regressor
2. Extra Trees Regressor
3. Hist Gradient Boosting Regressor

The best model is selected based on the lowest RMSE value.

## ✨ Main Features

### 1. Dataset Upload
- Upload electricity consumption datasets.
- Supports CSV and Excel files.
- Displays the uploaded dataset preview.
- Automatically detects date and consumption columns where possible.

### 2. Automatic Data Preprocessing
- Converts date columns into datetime format.
- Handles missing values.
- Removes invalid records.
- Sorts records chronologically.
- Converts data into hourly frequency.

### 3. Feature Engineering
Creates useful features such as:
- Hour
- Day
- Month
- Day of week
- Weekend indicator
- Lag features
- Rolling mean
- Cyclical time features

### 4. Model Training
- Trains multiple Machine Learning models.
- Compares model performance.
- Selects the best model automatically.
- Saves the trained model as a PKL file.

### 5. Electricity Consumption Prediction
- User enters the required number of forecast hours.
- Generates future electricity consumption predictions.
- Displays total forecast consumption.
- Generates future consumption graphs.

### 6. Model Evaluation
The application calculates:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

It also displays an Actual vs Predicted Consumption graph.

### 7. Download Results
- Download future electricity forecasts.
- Download model performance comparison.
- Save trained model files.

## 📂 Project Structure

```text
Electricity-Consumption-Prediction/
│
├── app/
│   └── main.py
│
├── dataset/
│
├── model/
│   ├── best_electricity_model.pkl
│   └── model_comparison.csv
│
├── prediction/
│
├── preprocessing/
│
├── venv/
│
└── README.md
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd Electricity-Consumption-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Required Libraries

```bash
python -m pip install streamlit pandas numpy scikit-learn joblib openpyxl
```

### Step 6: Run the Application

```bash
python -m streamlit run app/main.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🔄 Project Workflow

```text
Upload Electricity Dataset
          |
          v
Automatic Data Preprocessing
          |
          v
Feature Engineering
          |
          v
Train Machine Learning Models
          |
          v
Compare Model Performance
          |
          v
Select Best Performing Model
          |
          v
Save Model as PKL File
          |
          v
Enter Forecast Hours
          |
          v
Generate Electricity Predictions
          |
          v
Display Graphs and Results
          |
          v
Download Forecast CSV
```

## 📊 Model Evaluation Metrics

**MAE:** Measures the average absolute difference between actual and predicted values.

**RMSE:** Measures prediction errors while giving greater importance to larger errors.

**R² Score:** Measures how much variation in the target variable is explained by the model.

## 💾 Model Storage

The best-performing trained model is automatically saved as:

```text
model/best_electricity_model.pkl
```

Model performance comparison is saved as:

```text
model/model_comparison.csv
```

## 📈 Forecasting

Users can enter the number of hours they want to predict.

Examples:

- 24 hours – Next day forecast
- 48 hours – Next two days
- 168 hours – Next seven days
- 720 hours – Approximately 30 days

The application generates forecast values and displays a consumption graph.

## ⚠️ Current Limitations

- The current forecasting implementation is designed for hourly datasets.
- At least 200 valid hourly observations are recommended.
- The maximum forecast horizon is 720 hours.
- Forecast accuracy depends on dataset quality and historical patterns.
- Future forecasts are recursive, meaning earlier predictions are used to generate later predictions.
- Monthly totals are calculated from the generated hourly forecast period and may represent partial calendar months.

## 🚀 Future Enhancements

- Support daily and monthly input datasets.
- Improve prediction accuracy through hyperparameter tuning.
- Add advanced forecasting algorithms.
- Develop a detailed monthly analytics dashboard.
- Add actual versus predicted consumption reports.
- Deploy the application online.

## 👨‍💻 Conclusion

The Electricity Consumption Prediction system demonstrates how Machine Learning can be used to analyze historical electricity consumption and forecast future usage.

By automating data preprocessing, model training, model selection and prediction, the application provides a simple interface for electricity consumption forecasting.

The project helps users understand electricity usage patterns and explore future consumption estimates using Machine Learning.
