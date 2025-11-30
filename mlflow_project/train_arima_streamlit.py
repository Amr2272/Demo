import streamlit as st
import subprocess
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pickle
from itertools import product

st.set_page_config(page_title="ARIMA Trainer", layout="wide")
st.title("📊 ARIMA Trainer with MLflow, DVC & Model Registry")

# Paths and DVC
dataset_dvc = r"D:\Final Project\model_dataset.csv.dvc"
arima_model_dvc = r"D:\Final Project\mlflow_project\models\models.dvc"
dataset_path = os.path.splitext(dataset_dvc)[0]
arima_model_path = r"D:\Final Project\mlflow_project\models\arima_model.pkl"

mlflow_db_uri = "sqlite:///D:/Final_Project/mlflow_project/mlflow.db"
mlflow_artifact_root = "file:///D:/Final_Project/mlflow_project/mlruns"

MODEL_REGISTRY_NAME = "ARIMAForecastModel"

# 1️⃣ Pull dataset and model from DVC
st.subheader("1️⃣ Pull dataset and pre-trained model from DVC")
if st.button("Pull from DVC"):
    try:
        subprocess.run(["dvc", "pull", dataset_dvc], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        subprocess.run(["dvc", "pull", arima_model_dvc], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        st.success("✅ DVC files pulled successfully!")
    except Exception as e:
        st.error(f"DVC Error: {e}")

# 2️⃣ MLflow Setup
st.subheader("2️⃣ MLflow Setup")
experiment_name = "arima_test"
mlflow.set_tracking_uri(mlflow_db_uri)
mlflow.set_experiment(experiment_name)
st.write(f"MLflow Experiment: **{experiment_name}**")
st.write(f"Artifact root: {mlflow_artifact_root}")

# 3️⃣ Load Dataset
st.subheader("3️⃣ Load Dataset")
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    st.dataframe(df.head())
    # ARIMA expects a single time series column, e.g., "sales"
    series = df.set_index("date")["sales"]
    input_example = series.head(10).to_frame()

    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]
else:
    st.error(f"Dataset not found: {dataset_path}")
    st.stop()

# 4️⃣ Load Pre-trained ARIMA Model (Optional)
st.subheader("4️⃣ Load Pre-trained Model (Optional)")
pre_trained_model = None
if os.path.exists(arima_model_path):
    with open(arima_model_path, "rb") as f:
        pre_trained_model = pickle.load(f)
    st.success("✅ Pre-trained ARIMA model loaded locally")

# 5️⃣ ARIMA Wrapper for MLflow
class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        # model_input is a DataFrame with date index
        start_idx = model_input.index[0]
        end_idx = model_input.index[-1]
        return self.model.predict(start=start_idx, end=end_idx)

# 6️⃣ Train ARIMA (Grid Search)
st.subheader("5️⃣ Train ARIMA Model (Grid Search)")
param_grid = {"p": [1, 2], "d": [0, 1], "q": [0, 1]}

if st.button("Start Grid Search"):
    best_rmse = float("inf")
    best_mae = float("inf")
    best_params = None
    best_model_obj = None

    progress_text = st.empty()
    progress_bar = st.progress(0)

    grid = list(product(param_grid["p"], param_grid["d"], param_grid["q"]))
    total_runs = len(grid)
    run_count = 0

    for p, d, q in grid:
        run_name = f"ARIMA_{p}{d}{q}"
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"p": p, "d": d, "q": q})

                model = ARIMA(train_series, order=(p, d, q))
                model_fit = model.fit()

                y_train_pred = model_fit.predict(start=train_series.index[0], end=train_series.index[-1])
                y_test_pred = model_fit.predict(start=test_series.index[0], end=test_series.index[-1])

                train_rmse = mean_squared_error(train_series, y_train_pred) ** 0.5
                train_mae = mean_absolute_error(train_series, y_train_pred)
                test_rmse = mean_squared_error(test_series, y_test_pred) ** 0.5
                test_mae = mean_absolute_error(test_series, y_test_pred)

                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("test_mae", test_mae)

                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_mae = test_mae
                    best_params = {"p": p, "d": d, "q": q}
                    best_model_obj = model_fit

                run_count += 1
                progress_text.text(f"Run {run_count}/{total_runs} — MAE={test_mae:.2f}, RMSE={test_rmse:.2f}")
                progress_bar.progress(run_count / total_runs)

        except Exception as e:
            st.error(f"Run {run_name} failed: {e}")

    st.success(f"🎉 Grid search complete. Best RMSE={best_rmse:.2f}, Best MAE={best_mae:.2f}")
    st.write("Best Parameters:", best_params)

    # 7️⃣ Register Best Model
    if best_model_obj is not None:
        os.makedirs(os.path.dirname(arima_model_path), exist_ok=True)
        with open(arima_model_path, "wb") as f:
            pickle.dump(best_model_obj, f)

        wrapper_model = ARIMAModelWrapper(best_model_obj)
        signature = infer_signature(input_example, best_model_obj.predict(start=input_example.index[0], end=input_example.index[-1]))

        with mlflow.start_run(run_name="Best_Model_Final"):
            mlflow.log_params(best_params)
            mlflow.log_metric("rmse", best_rmse)
            mlflow.log_metric("mae", best_mae)
            mlflow.pyfunc.log_model(
                artifact_path="arima_model",
                python_model=wrapper_model,
                registered_model_name=MODEL_REGISTRY_NAME,
                signature=signature,
                input_example=input_example
            )

        st.success(f"🏆 Best ARIMA model registered to MLflow")

        try:
            subprocess.run(["dvc", "add", arima_model_path], check=True)
            subprocess.run(["dvc", "push"], check=True)
            st.success("📦 DVC updated and pushed!")
        except Exception as e:
            st.warning(f"DVC Push failed: {e}")

# 8️⃣ MLflow Model Registry Management
st.subheader("7️⃣ MLflow Model Registry Management")
version = st.text_input("Enter model version to promote:", "1")
col1, col2 = st.columns(2)
if col1.button("Promote to STAGING"):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name=MODEL_REGISTRY_NAME, version=int(version), stage="Staging")
    st.success(f"✔️ Model v{version} promoted to **STAGING**")
if col2.button("Promote to PRODUCTION"):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name=MODEL_REGISTRY_NAME, version=int(version), stage="Production")
    st.success(f"🚀 Model v{version} promoted to **PRODUCTION**")

# 9️⃣ MLflow UI
st.subheader("8️⃣ MLflow UI")
st.write("👉 Open in browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)")
if st.button("Launch MLflow UI"):
    mlflow_ui_command = [
        "mlflow", "ui",
        "--backend-store-uri", mlflow_db_uri,
        "--default-artifact-root", mlflow_artifact_root,
        "--host", "127.0.0.1",
        "--port", "5000"
    ]
    subprocess.Popen(mlflow_ui_command)
    st.success("🎉 MLflow UI launched!")
    
# streamlit run "D:\Final Project\mlflow_project\train_arima_streamlit.py"