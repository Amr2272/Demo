import streamlit as st
import subprocess
import os
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.prophet
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pickle
from itertools import product

st.set_page_config(page_title="Forecast Model Trainer", layout="wide")
st.title("📊 Prophet & ARIMA Trainer with MLflow, DVC & Model Registry")

# Paths and DVC
dataset_dvc = r"D:\Final Project\mlflow_project\data\model_dataset.csv.dvc"
prophet_model_dvc = r"D:\Final Project\mlflow_project\models\models.dvc"
arima_model_dvc = r"D:\Final Project\mlflow_project\models\models.dvc"
dataset_path = os.path.splitext(dataset_dvc)[0]
prophet_model_path = r"D:\Final Project\mlflow_project\models\models\prophet_tuned_model.pkl"
arima_model_path = r"D:\Final Project\mlflow_project\models\models_arima\arima_model.pkl"

mlflow_db_uri = "sqlite:///D:/Final_Project/mlflow_project/mlflow.db"
mlflow_artifact_root = "file:///D:/Final_Project/mlflow_project/mlruns"

PROPHET_REGISTRY_NAME = "ProphetForecastModel"
ARIMA_REGISTRY_NAME = "ARIMAForecastModel"
BEST_MODELS_EXPERIMENT = "best_models"

# Initialize MLflow
mlflow.set_tracking_uri(mlflow_db_uri)

# ARIMA Wrapper for MLflow
class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        start_idx = model_input.index[0]
        end_idx = model_input.index[-1]
        return self.model.predict(start=start_idx, end=end_idx)

# 1️⃣ Pull dataset and models from DVC
st.subheader("1️⃣ Pull dataset and pre-trained models from DVC")
if st.button("Pull from DVC"):
    try:
        subprocess.run(["dvc", "pull", dataset_dvc], check=True)
        subprocess.run(["dvc", "pull", prophet_model_dvc], check=True)
        subprocess.run(["dvc", "pull", arima_model_dvc], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        st.success("✅ DVC files pulled successfully!")
    except Exception as e:
        st.error(f"DVC Error: {e}")

# 2️⃣ Load Dataset
st.subheader("2️⃣ Load Dataset")
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Prepare data for Prophet
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
    prophet_input_example = prophet_df.head(5)
    
    # Prepare data for ARIMA
    series = df.set_index("date")["sales"]
    arima_input_example = series.head(10).to_frame()
    
    # Train/Test Split for Prophet
    prophet_train_df, prophet_test_df = train_test_split(prophet_df, test_size=0.2, shuffle=False)
    
    # Train/Test Split for ARIMA
    train_size = int(len(series) * 0.8)
    arima_train_series = series[:train_size]
    arima_test_series = series[train_size:]
    
    st.write(f"Training rows: {len(prophet_train_df)}, Testing rows: {len(prophet_test_df)}")
else:
    st.error(f"Dataset not found: {dataset_path}")
    st.stop()

# 3️⃣ Load Pre-trained Models (Optional)
st.subheader("3️⃣ Load Pre-trained Models (Optional)")
col1, col2 = st.columns(2)

with col1:
    prophet_pre_trained = None
    if os.path.exists(prophet_model_path):
        with open(prophet_model_path, "rb") as f:
            prophet_pre_trained = pickle.load(f)
        st.success("✅ Pre-trained Prophet model loaded")

with col2:
    arima_pre_trained = None
    if os.path.exists(arima_model_path):
        with open(arima_model_path, "rb") as f:
            arima_pre_trained = pickle.load(f)
        st.success("✅ Pre-trained ARIMA model loaded")

# 4️⃣ Prophet Model Training
st.subheader("4️⃣ Prophet Model Training")
prophet_param_grid = {
    "changepoint_prior_scale": [0.001],
    "seasonality_prior_scale": [0.1],
    "holidays_prior_scale": [0.1],
    "seasonality_mode": ["additive"],
    "yearly_seasonality": [True],
    "weekly_seasonality": [True]
}

if st.button("Train Prophet Model"):
    st.write("🚀 Starting Prophet Grid Search...")
    
    # Set up Prophet experiment
    mlflow.set_experiment("prophet_test")
    
    best_prophet_rmse = float("inf")
    best_prophet_mae = float("inf")
    best_prophet_params = None
    best_prophet_model = None

    progress_text = st.empty()
    progress_bar = st.progress(0)

    total_runs = len(prophet_param_grid["changepoint_prior_scale"]) * \
                 len(prophet_param_grid["seasonality_prior_scale"]) * \
                 len(prophet_param_grid["holidays_prior_scale"]) * \
                 len(prophet_param_grid["seasonality_mode"]) * \
                 len(prophet_param_grid["yearly_seasonality"]) * \
                 len(prophet_param_grid["weekly_seasonality"])
    run_count = 0

    for cps, sps, hps, mode, yearly, weekly in product(
        prophet_param_grid["changepoint_prior_scale"],
        prophet_param_grid["seasonality_prior_scale"],
        prophet_param_grid["holidays_prior_scale"],
        prophet_param_grid["seasonality_mode"],
        prophet_param_grid["yearly_seasonality"],
        prophet_param_grid["weekly_seasonality"]
    ):
        run_name = f"Prophet_cps{cps}_sps{sps}_hps{hps}_mode{mode}"
        params = {
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
            "holidays_prior_scale": hps,
            "seasonality_mode": mode,
            "yearly_seasonality": yearly,
            "weekly_seasonality": weekly,
            "daily_seasonality": False
        }

        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)

                model = Prophet(**params)
                model.fit(prophet_train_df)

                # Forecast
                forecast_train = model.predict(prophet_train_df)
                y_train_pred = forecast_train["yhat"]
                train_rmse = mean_squared_error(prophet_train_df["y"], y_train_pred) ** 0.5
                train_mae = mean_absolute_error(prophet_train_df["y"], y_train_pred)

                forecast_test = model.predict(prophet_test_df)
                y_test_pred = forecast_test["yhat"]
                test_rmse = mean_squared_error(prophet_test_df["y"], y_test_pred) ** 0.5
                test_mae = mean_absolute_error(prophet_test_df["y"], y_test_pred)

                # Log metrics
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("test_mae", test_mae)

                if test_rmse < best_prophet_rmse:
                    best_prophet_rmse = test_rmse
                    best_prophet_mae = test_mae
                    best_prophet_params = params
                    best_prophet_model = model

                run_count += 1
                progress_text.text(f"Prophet Run {run_count}/{total_runs} — MAE={test_mae:.2f}, RMSE={test_rmse:.2f}")
                progress_bar.progress(run_count / total_runs)

        except Exception as e:
            st.error(f"Prophet Run {run_name} failed: {e}")

    st.success(f"🎉 Prophet Grid search complete. Best RMSE={best_prophet_rmse:.2f}, Best MAE={best_prophet_mae:.2f}")
    st.write("Best Prophet Parameters:", best_prophet_params)

    # Save and register best Prophet model
    if best_prophet_model is not None:
        # Save locally
        os.makedirs(os.path.dirname(prophet_model_path), exist_ok=True)
        with open(prophet_model_path, "wb") as f:
            pickle.dump(best_prophet_model, f)

        # Register to best_models experiment
        mlflow.set_experiment(BEST_MODELS_EXPERIMENT)
        with mlflow.start_run(run_name="Best_Prophet_Model"):
            mlflow.log_params(best_prophet_params)
            mlflow.log_metric("rmse", best_prophet_rmse)
            mlflow.log_metric("mae", best_prophet_mae)
            mlflow.set_tag("model_type", "prophet")
            
            # Generate Signature and log model
            prediction_signature = best_prophet_model.predict(prophet_input_example)
            signature = infer_signature(prophet_input_example, prediction_signature)

            mlflow.prophet.log_model(
                best_prophet_model,
                artifact_path="prophet_model",
                registered_model_name=PROPHET_REGISTRY_NAME,
                input_example=prophet_input_example,
                signature=signature
            )

        st.success(f"🏆 Best Prophet Model registered to MLflow (RMSE: {best_prophet_rmse:.2f})")

        # DVC update
        try:
            subprocess.run(["dvc", "add", prophet_model_path], check=True)
            subprocess.run(["dvc", "push"], check=True)
            st.success("📦 Prophet model updated in DVC!")
        except Exception as e:
            st.warning(f"Prophet DVC Push failed: {e}")

# 5️⃣ ARIMA Model Training
st.subheader("5️⃣ ARIMA Model Training")
arima_param_grid = {"p": [1, 2], "d": [0, 1], "q": [0, 1]}

if st.button("Train ARIMA Model"):
    st.write("🚀 Starting ARIMA Grid Search...")
    
    # Set up ARIMA experiment
    mlflow.set_experiment("arima_test")
    
    best_arima_rmse = float("inf")
    best_arima_mae = float("inf")
    best_arima_params = None
    best_arima_model = None

    progress_text = st.empty()
    progress_bar = st.progress(0)

    grid = list(product(arima_param_grid["p"], arima_param_grid["d"], arima_param_grid["q"]))
    total_runs = len(grid)
    run_count = 0

    for p, d, q in grid:
        run_name = f"ARIMA_{p}{d}{q}"
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"p": p, "d": d, "q": q})

                model = ARIMA(arima_train_series, order=(p, d, q))
                model_fit = model.fit()

                y_train_pred = model_fit.predict(start=arima_train_series.index[0], end=arima_train_series.index[-1])
                y_test_pred = model_fit.predict(start=arima_test_series.index[0], end=arima_test_series.index[-1])

                train_rmse = mean_squared_error(arima_train_series, y_train_pred) ** 0.5
                train_mae = mean_absolute_error(arima_train_series, y_train_pred)
                test_rmse = mean_squared_error(arima_test_series, y_test_pred) ** 0.5
                test_mae = mean_absolute_error(arima_test_series, y_test_pred)

                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("test_mae", test_mae)

                if test_rmse < best_arima_rmse:
                    best_arima_rmse = test_rmse
                    best_arima_mae = test_mae
                    best_arima_params = {"p": p, "d": d, "q": q}
                    best_arima_model = model_fit

                run_count += 1
                progress_text.text(f"ARIMA Run {run_count}/{total_runs} — MAE={test_mae:.2f}, RMSE={test_rmse:.2f}")
                progress_bar.progress(run_count / total_runs)

        except Exception as e:
            st.error(f"ARIMA Run {run_name} failed: {e}")

    st.success(f"🎉 ARIMA Grid search complete. Best RMSE={best_arima_rmse:.2f}, Best MAE={best_arima_mae:.2f}")
    st.write("Best ARIMA Parameters:", best_arima_params)

    # Save and register best ARIMA model
    if best_arima_model is not None:
        os.makedirs(os.path.dirname(arima_model_path), exist_ok=True)
        with open(arima_model_path, "wb") as f:
            pickle.dump(best_arima_model, f)

        wrapper_model = ARIMAModelWrapper(best_arima_model)
        signature = infer_signature(arima_input_example, best_arima_model.predict(start=arima_input_example.index[0], end=arima_input_example.index[-1]))

        # Register to best_models experiment
        mlflow.set_experiment(BEST_MODELS_EXPERIMENT)
        with mlflow.start_run(run_name="Best_ARIMA_Model"):
            mlflow.log_params(best_arima_params)
            mlflow.log_metric("rmse", best_arima_rmse)
            mlflow.log_metric("mae", best_arima_mae)
            mlflow.set_tag("model_type", "arima")
            
            mlflow.pyfunc.log_model(
                artifact_path="arima_model",
                python_model=wrapper_model,
                registered_model_name=ARIMA_REGISTRY_NAME,
                signature=signature,
                input_example=arima_input_example
            )

        st.success(f"🏆 Best ARIMA model registered to MLflow (RMSE: {best_arima_rmse:.2f})")

        try:
            subprocess.run(["dvc", "add", arima_model_path], check=True)
            subprocess.run(["dvc", "push"], check=True)
            st.success("📦 ARIMA model updated in DVC!")
        except Exception as e:
            st.warning(f"ARIMA DVC Push failed: {e}")

# 6️⃣ Model Registry Management
st.subheader("6️⃣ Model Registry Management")
st.write("Promote models to different stages:")

col1, col2 = st.columns(2)

with col1:
    st.write("**Prophet Model**")
    prophet_version = st.text_input("Prophet version to promote:", "1")
    if st.button("Promote Prophet to STAGING"):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=PROPHET_REGISTRY_NAME, version=int(prophet_version), stage="Staging")
        st.success(f"✔️ Prophet v{prophet_version} promoted to **STAGING**")
    if st.button("Promote Prophet to PRODUCTION"):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=PROPHET_REGISTRY_NAME, version=int(prophet_version), stage="Production")
        st.success(f"🚀 Prophet v{prophet_version} promoted to **PRODUCTION**")

with col2:
    st.write("**ARIMA Model**")
    arima_version = st.text_input("ARIMA version to promote:", "1")
    if st.button("Promote ARIMA to STAGING"):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=ARIMA_REGISTRY_NAME, version=int(arima_version), stage="Staging")
        st.success(f"✔️ ARIMA v{arima_version} promoted to **STAGING**")
    if st.button("Promote ARIMA to PRODUCTION"):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=ARIMA_REGISTRY_NAME, version=int(arima_version), stage="Production")
        st.success(f"🚀 ARIMA v{arima_version} promoted to **PRODUCTION**")

# 7️⃣ MLflow UI
st.subheader("7️⃣ MLflow UI")
st.write("👉 Open in browser: http://127.0.0.1:5000")
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

# 8️⃣ Experiment Status
st.subheader("8️⃣ Experiment Status")
if st.button("Check Best Models Experiment"):
    client = mlflow.tracking.MlflowClient()
    try:
        experiment = client.get_experiment_by_name(BEST_MODELS_EXPERIMENT)
        if experiment:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            st.write(f"**Best Models Experiment:** {BEST_MODELS_EXPERIMENT}")
            st.write(f"**Total Best Model Runs:** {len(runs)}")
            
            if runs:
                st.write("**Registered Best Models:**")
                for run in runs:
                    model_type = run.data.tags.get("model_type", "unknown")
                    rmse = run.data.metrics.get("rmse", "N/A")
                    st.write(f"- {run.info.run_name} ({model_type}) - RMSE: {rmse}")
        else:
            st.info("Best Models experiment not created yet. Train models first.")
    except Exception as e:
        st.error(f"Error checking experiment status: {e}")

# streamlit run "D:\Final Project\mlflow_project\train_models_streamlit.py"         