# End-to-End MLOps Pipeline for Predictive Maintenance

## Project Summary
This repository contains a comprehensive, end-to-end MLOps pipeline for predicting equipment failure. The project demonstrates a full data science lifecycle: from baseline modeling to a comparative analysis of tabular (XGBoost) and time-series (LSTM/GRU) models. All experiments are tracked using **MLflow**, and the champion model is served via an interactive **Streamlit** dashboard.

## Core Features
* **Comparative Model Analysis:** Systematically evaluates four different models (Random Forest, XGBoost, LSTM, GRU) across two distinct datasets (AI4I 2020 and NASA Turbofan) to identify the optimal architecture.
* **Time-Series Deep Learning:** Implements and trains LSTM and GRU networks on the NASA Turbofan dataset, transforming raw sensor data into 3D "windowed" sequences to capture temporal patterns.
* **MLOps Integration:** Utilizes **MLflow** for robust experiment tracking. A master script (`train.py`) logs all parameters, metrics, and model artifacts, ensuring full reproducibility and easy model comparison.
* **Interactive Deployment:** Includes a production-ready **Streamlit** dashboard (`app.py`) that loads the champion model directly from the MLflow run. The UI demonstrates the model's capabilities on unseen test data.

## Modeling Results: Performance Comparison
The primary objective was to maximize the **Recall** for the failure class (Class 1), representing the model's ability to "catch" an impending failure. The time-series models demonstrated a significant performance improvement over the tabular "snapshot" models.

| Model | Dataset | Model Type | Recall (Class 1) |
| :--- | :--- | :--- | :--- |
| RandomForest | AI4I 2020 | Tabular | `~0.44` (Baseline) |
| XGBoost | AI4I 2020 | Tabular | `~0.77` |
| GRU | NASA Turbofan | Time-Series | `~0.97` |
| **LSTM** | **NASA Turbofan** | **Time-Series** | **`~0.98` (Champion)** |

## Technical Architecture

### 1. Training & Experimentation (`train.py`)
This script serves as the master pipeline for all model training and logging. When executed, it sequentially:
1.  Sets up the MLflow experiment and tracking URI (`sqlite:///mlflow.db`).
2.  **Pipeline 1 (Tabular):** Loads and preprocesses the AI4I 2020 dataset.
3.  Trains and logs a Random Forest baseline model.
4.  Trains and logs an optimized XGBoost model.
5.  **Pipeline 2 (Time-Series):** Loads, preprocesses, and "windows" the NASA Turbofan dataset.
6.  Trains and logs the LSTM model.
7.  Trains and logs the GRU model.

All results are saved to the `mlflow.db` database and the `mlruns` artifact store.

### 2. Deployment & Visualization (`app.py`)
This script builds the user-facing web application.
1.  Connects to the MLflow tracking URI (`sqlite:///mlflow.db`).
2.  Loads the champion model (specified by its `Run ID`) directly from the `mlruns` artifact store using `mlflow.pyfunc.load_model`.
3.  Loads the `test_FD001.txt` data to populate the UI.
4.  Provides selectors for a user to choose an engine and a specific cycle.
5.  Performs all necessary real-time data transformation (scaling, windowing) to create the 3D tensor required by the model.
6.  Feeds the tensor to the model and displays the resulting prediction (Normal/Failure) and failure probability.

---

## Usage Instructions
To replicate this project, follow these steps:

### 1. Download Datasets
Two datasets are required. Download and place them in the root of this project folder:
* **AI4I Data:** `ai4i2020.csv` (from the [UCI Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset))
* **NASA Data:** `train_FD001.txt` and `test_FD001.txt` (from [this Kaggle Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps))

### 2. Install Dependencies
Install all required Python libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 3. Run the Master Training Pipeline
This will train all four models and generate the `mlflow.db` and `mlruns` directories. This process will take several minutes due to the deep learning models.
```bash
python train.py
```

### 4. Select Best Model
Launch the MLflow UI to compare your models and find the `Run ID` for your best-performing model (e.g., the `LSTM` run).
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000`, open the experiment, click the champion model's run, and copy its **Run ID**.

### 5. Launch the Streamlit Application
1.  Open `app.py` in your code editor.
2.  Paste your champion **Run ID** into the `YOUR_RUN_ID = "..."` variable at the top of the script.
3.  Save the file and run the app from your terminal:
```bash
streamlit run app.py
```
---
