import streamlit as st
import subprocess
import os
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.prophet
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pickle
from itertools import product

st.set_page_config(page_title="Prophet Forecast Trainer", layout="wide")
st.title("📊 Prophet Forecast Trainer with MLflow, DVC & Model Registry")

# Paths and DVC
dataset_dvc = r"D:\Final Project\model_dataset.csv.dvc"
prophet_model_dvc = r"D:\Final Project\mlflow_project\models\models.dvc"
dataset_path = os.path.splitext(dataset_dvc)[0]
prophet_model_path = r"D:\Final Project\mlflow_project\models\models\prophet_tuned_model.pkl"

mlflow_db_uri = "sqlite:///D:/Final_Project/mlflow_project/mlflow.db"
mlflow_artifact_root = "file:///D:/Final_Project/mlflow_project/mlruns"


MODEL_REGISTRY_NAME = "ProphetForecastModel"

# 1️⃣ Pull dataset and model from DVC
st.subheader("1️⃣ Pull dataset and pre-trained model from DVC")
if st.button("Pull from DVC"):
    st.text(f"Pulling dataset from DVC: {dataset_dvc}")
    try:
        subprocess.run(["dvc", "pull", dataset_dvc], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        
        st.text(f"Pulling Prophet model from DVC: {prophet_model_dvc}")
        subprocess.run(["dvc", "pull", prophet_model_dvc], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        st.success("✅ DVC files pulled successfully!")
    except Exception as e:
        st.error(f"DVC Error: {e}")

# 2️⃣ MLflow Setup
st.subheader("2️⃣ MLflow Setup")

# Local MLflow SQLite database
mlflow_db_uri = "sqlite:///D:/Final_Project/mlflow_project/mlflow.db"

# Local artifact storage
mlflow_artifact_root = "file:///D:/Final_Project/mlflow_project/mlruns"

# NEW experiment name
experiment_name = "prophet_test"

# Set tracking URI
mlflow.set_tracking_uri(mlflow_db_uri)

# Set or create experiment
mlflow.set_experiment(experiment_name)

st.write(f"MLflow Experiment: **{experiment_name}**")
st.write(f"Artifact root: {mlflow_artifact_root}")


# 3️⃣ Load Dataset
st.subheader("3️⃣ Load Dataset")
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
    input_example = prophet_df.head(5)

    # Train/Test Split
    train_df, test_df = train_test_split(prophet_df, test_size=0.2, shuffle=False)
    st.write(f"Training rows: {len(train_df)}, Testing rows: {len(test_df)}")
else:
    st.error(f"Dataset not found: {dataset_path}")
    st.stop()

# 4️⃣ Load Pre-trained Model Locally (Optional)
st.subheader("4️⃣ Load Pre-trained Model (Optional)")
pre_trained_model = None
if os.path.exists(prophet_model_path):
    with open(prophet_model_path, "rb") as f:
        pre_trained_model = pickle.load(f)
    st.success("✅ Pre-trained model loaded locally (not logged to MLflow)")

# 5️⃣ Train Prophet Model (Grid Search)
st.subheader("5️⃣ Train Prophet Model (Grid Search)")
param_grid = {
    "changepoint_prior_scale": [0.001],
    "seasonality_prior_scale": [0.1],
    "holidays_prior_scale": [0.1],
    "seasonality_mode": ["additive"],
    "yearly_seasonality": [True, False],
    "weekly_seasonality": [True, False]
}

if st.button("Start Grid Search"):
    best_rmse = float("inf")
    best_mae = float("inf")
    best_params = None
    best_model_obj = None

    progress_text = st.empty()
    progress_bar = st.progress(0)

    total_runs = len(param_grid["changepoint_prior_scale"]) * \
                 len(param_grid["seasonality_prior_scale"]) * \
                 len(param_grid["holidays_prior_scale"]) * \
                 len(param_grid["seasonality_mode"]) * \
                 len(param_grid["yearly_seasonality"]) * \
                 len(param_grid["weekly_seasonality"])
    run_count = 0

    for cps, sps, hps, mode, yearly, weekly in product(
        param_grid["changepoint_prior_scale"],
        param_grid["seasonality_prior_scale"],
        param_grid["holidays_prior_scale"],
        param_grid["seasonality_mode"],
        param_grid["yearly_seasonality"],
        param_grid["weekly_seasonality"]
    ):
        run_name = f"cps{cps}_sps{sps}_hps{hps}_mode{mode}"
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
                model.fit(train_df)

                # Forecast
                forecast_train = model.predict(train_df)
                y_train_pred = forecast_train["yhat"]
                train_rmse = mean_squared_error(train_df["y"], y_train_pred) ** 0.5
                train_mae = mean_absolute_error(train_df["y"], y_train_pred)

                forecast_test = model.predict(test_df)
                y_test_pred = forecast_test["yhat"]
                test_rmse = mean_squared_error(test_df["y"], y_test_pred) ** 0.5
                test_mae = mean_absolute_error(test_df["y"], y_test_pred)

                # Log metrics
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("test_mae", test_mae)

                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_mae = test_mae
                    best_params = params
                    best_model_obj = model 

                run_count += 1
                # --- UPDATED DISPLAY TEXT ---
                progress_text.text(f"Run {run_count}/{total_runs} — MAE={test_mae:.2f}, RMSE={test_rmse:.2f}")
                progress_bar.progress(run_count / total_runs)

        except Exception as e:
            st.error(f"Run {run_name} failed: {e}")

    st.success(f"🎉 Grid search complete. Best RMSE={best_rmse:.2f}, Best MAE={best_mae:.2f}")
    st.write("Best Parameters:", best_params)

    # 6️⃣ Register ONLY the Best Model
    if best_model_obj is not None:
        # Save locally
        os.makedirs(os.path.dirname(prophet_model_path), exist_ok=True)
        with open(prophet_model_path, "wb") as f:
            pickle.dump(best_model_obj, f)

        # Log to MLflow
        with mlflow.start_run(run_name="Best_Model_Final"):
            mlflow.log_params(best_params)
            mlflow.log_metric("rmse", best_rmse)
            mlflow.log_metric("mae", best_mae)
            
            # Generate Signature
            prediction_signature = best_model_obj.predict(input_example)
            signature = infer_signature(input_example, prediction_signature)

            mlflow.prophet.log_model(
                best_model_obj,
                name="prophet_model",
                registered_model_name=MODEL_REGISTRY_NAME,
                input_example=input_example,
                signature=signature
            )

        st.success(f"🏆 Only the Best Model (RMSE: {best_rmse:.2f}, MAE: {best_mae:.2f}) was registered to MLflow.")

        # DVC update
        try:
            subprocess.run(["dvc", "add", prophet_model_path], check=True)
            subprocess.run(["dvc", "push"], check=True)
            st.success("📦 DVC updated and pushed!")
        except Exception as e:
            st.warning(f"DVC Push failed: {e}")

# 7️⃣ MLflow Model Registry Management
st.subheader("7️⃣ MLflow Model Registry Management")
version = st.text_input("Enter model version to promote:", "1")
col1, col2 = st.columns(2)
if col1.button("Promote to STAGING"):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=int(version),
        stage="Staging"
    )
    st.success(f"✔️ Model v{version} promoted to **STAGING**")
if col2.button("Promote to PRODUCTION"):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=int(version),
        stage="Production"
    )
    st.success(f"🚀 Model v{version} promoted to **PRODUCTION**")

# 8️⃣ MLflow UI
st.subheader("8️⃣ MLflow UI")
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

# streamlit run "D:\Final Project\mlflow_project\train_prophet_streamlit.py"

















# import streamlit as st
# import subprocess
# import os
# import pandas as pd
# from prophet import Prophet
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
# import mlflow
# import mlflow.prophet
# import mlflow.pyfunc
# import pickle
# from itertools import product

# st.set_page_config(page_title="Prophet Forecast Trainer", layout="wide")
# st.title("📊 Prophet Forecast Trainer with MLflow, DVC & Model Registry")

# # Paths and DVC
# dataset_dvc = r"D:\Final Project\mlflow_project\data\model_dataset.csv.dvc"
# prophet_model_dvc = r"D:\Final Project\mlflow_project\models\models.dvc"
# dataset_path = os.path.splitext(dataset_dvc)[0]
# prophet_model_path = r"D:\Final Project\mlflow_project\models\models\prophet_tuned_model.pkl"

# mlflow_db_uri = "sqlite:///D:/Final_Project/mlflow_project/mlflow.db"
# mlflow_artifact_root = r"D:\Final Project\mlflow_project\mlruns"

# MODEL_REGISTRY_NAME = "ProphetForecastModel"

# # 1️⃣ Pull dataset and model from DVC
# st.subheader("1️⃣ Pull dataset and pre-trained model from DVC")
# if st.button("Pull from DVC"):
#     st.text(f"Pulling dataset from DVC: {dataset_dvc}")
#     subprocess.run(["dvc", "pull", dataset_dvc], check=True)
#     subprocess.run(["dvc", "checkout"], check=True)

#     st.text(f"Pulling Prophet model from DVC: {prophet_model_dvc}")
#     subprocess.run(["dvc", "pull", prophet_model_dvc], check=True)
#     subprocess.run(["dvc", "checkout"], check=True)
#     st.success("✅ DVC files pulled successfully!")

# # 2️⃣ MLflow Setup
# st.subheader("2️⃣ MLflow Setup")
# mlflow.set_tracking_uri(mlflow_db_uri)
# experiment_name = "sales_forecast_prophet"
# mlflow.set_experiment(experiment_name)
# st.write(f"MLflow Experiment: **{experiment_name}**")
# st.write(f"Artifact root: {mlflow_artifact_root}")

# # 3️⃣ Load Dataset
# st.subheader("3️⃣ Load Dataset")
# df = pd.read_csv(dataset_path, parse_dates=["date"])
# st.write("Dataset Preview:")
# st.dataframe(df.head())
# prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
# input_example = prophet_df.head(5)

# # Train/Test Split
# train_df, test_df = train_test_split(prophet_df, test_size=0.2, shuffle=False)
# st.write(f"Training rows: {len(train_df)}, Testing rows: {len(test_df)}")

# # 4️⃣ Load Pre-trained Model + AUTO REGISTER
# st.subheader("4️⃣ Load Pre-trained Model (Auto-Register)")

# pre_trained_model = None

# if os.path.exists(prophet_model_path):
#     with open(prophet_model_path, "rb") as f:
#         pre_trained_model = pickle.load(f)

#     st.success("✅ Pre-trained model loaded locally")

#     # -----------------------------------------------
#     # AUTO LOG + REGISTER PRETRAINED MODEL IN MLFLOW
#     # -----------------------------------------------
#     with mlflow.start_run(run_name="pretrained_model_autolog") as run:
#         mlflow.set_tags({
#             "model_type": "pretrained",
#             "source": "DVC",
#             "auto_registered": "true"
#         })

#         # Log parameters (if exist)
#         try:
#             mlflow.log_param("seasonality_mode", pre_trained_model.seasonality_mode)
#             mlflow.log_param("seasonality_prior_scale", pre_trained_model.seasonality_prior_scale)
#         except:
#             pass

#         # Log pretrained model file
#         mlflow.log_artifact(prophet_model_path)

#         # Log & Register the model
#         mlflow.prophet.log_model(
#             pre_trained_model,
#             artifact_path="pretrained_model",
#             registered_model_name=MODEL_REGISTRY_NAME,
#             input_example=input_example
#         )

#         st.success("📌 Pretrained model LOGGED & REGISTERED in MLflow")
#         st.write(f"✔ Run ID: {run.info.run_id}")

# # 5️⃣ Train Prophet Model (Grid Search)
# st.subheader("5️⃣ Train Prophet Model (Grid Search)")
# param_grid = {
#     "changepoint_prior_scale": [1],
#     "seasonality_prior_scale": [0.1],
#     "holidays_prior_scale": [0.1],
#     "seasonality_mode": ["additive"],
#     "yearly_seasonality": [True],
#     "weekly_seasonality": [True]
# }

# if st.button("Start Grid Search"):
#     best_rmse = float("inf")
#     best_params = None
#     best_model_obj = None

#     progress_text = st.empty()
#     progress_bar = st.progress(0)

#     total_runs = len(param_grid["changepoint_prior_scale"]) * \
#                  len(param_grid["seasonality_prior_scale"]) * \
#                  len(param_grid["holidays_prior_scale"]) * \
#                  len(param_grid["seasonality_mode"]) * \
#                  len(param_grid["yearly_seasonality"]) * \
#                  len(param_grid["weekly_seasonality"])
#     run_count = 0

#     for cps, sps, hps, mode, yearly, weekly in product(
#         param_grid["changepoint_prior_scale"],
#         param_grid["seasonality_prior_scale"],
#         param_grid["holidays_prior_scale"],
#         param_grid["seasonality_mode"],
#         param_grid["yearly_seasonality"],
#         param_grid["weekly_seasonality"]
#     ):
#         run_name = f"cps{cps}_sps{sps}_hps{hps}_mode{mode}"
#         params = {
#             "changepoint_prior_scale": cps,
#             "seasonality_prior_scale": sps,
#             "holidays_prior_scale": hps,
#             "seasonality_mode": mode,
#             "yearly_seasonality": yearly,
#             "weekly_seasonality": weekly,
#             "daily_seasonality": False
#         }

#         try:
#             with mlflow.start_run(run_name=run_name):
#                 mlflow.log_params(params)

#                 model = Prophet(**params)
#                 model.fit(train_df)

#                 forecast_train = model.predict(train_df)
#                 y_train_true = train_df["y"]
#                 y_train_pred = forecast_train["yhat"]
#                 train_rmse = mean_squared_error(y_train_true, y_train_pred) ** 0.5
#                 train_mae = mean_absolute_error(y_train_true, y_train_pred)

#                 forecast_test = model.predict(test_df)
#                 y_test_true = test_df["y"]
#                 y_test_pred = forecast_test["yhat"]
#                 test_rmse = mean_squared_error(y_test_true, y_test_pred) ** 0.5
#                 test_mae = mean_absolute_error(y_test_true, y_test_pred)

#                 mlflow.log_metric("train_rmse", train_rmse)
#                 mlflow.log_metric("train_mae", train_mae)
#                 mlflow.log_metric("test_rmse", test_rmse)
#                 mlflow.log_metric("test_mae", test_mae)

#                 mlflow.prophet.log_model(
#                     model,
#                     name="prophet_model",
#                     registered_model_name=MODEL_REGISTRY_NAME,
#                     input_example=input_example
#                 )

#                 if test_rmse < best_rmse:
#                     best_rmse = test_rmse
#                     best_params = params
#                     best_model_obj = model

#                 run_count += 1
#                 progress_text.text(f"Run {run_count}/{total_runs} — Train RMSE={train_rmse:.2f}, Test RMSE={test_rmse:.2f}")
#                 progress_bar.progress(run_count / total_runs)

#         except Exception as e:
#             st.error(f"Run {run_name} failed: {e}")

#     st.success(f"🎉 Grid search complete. Best Test RMSE={best_rmse:.2f}")
#     st.write("Best Parameters:", best_params)

#     # Save + Register best model
#     if best_model_obj is not None:
#         with open(prophet_model_path, "wb") as f:
#             pickle.dump(best_model_obj, f)

#         with mlflow.start_run(run_name="Best_Model_Registered"):
#             mlflow.log_params(best_params)
#             mlflow.log_metric("rmse", best_rmse)
#             mlflow.prophet.log_model(
#                 best_model_obj,
#                 name="prophet_model",
#                 registered_model_name=MODEL_REGISTRY_NAME,
#                 input_example=input_example
#             )

#         st.success("🏆 Best model registered in MLflow Model Registry")

#         subprocess.run(["dvc", "add", prophet_model_path], check=True)
#         subprocess.run(["dvc", "push"], check=True)
#         st.success("📦 DVC updated and pushed!")

# # 6️⃣ MLflow Model Registry Management
# st.subheader("7️⃣ MLflow Model Registry Management")
# version = st.text_input("Enter model version to promote:", "1")
# col1, col2 = st.columns(2)
# if col1.button("Promote to STAGING"):
#     client = mlflow.tracking.MlflowClient()
#     client.transition_model_version_stage(
#         name=MODEL_REGISTRY_NAME,
#         version=int(version),
#         stage="Staging"
#     )
#     st.success(f"✔️ Model v{version} promoted to **STAGING**")
# if col2.button("Promote to PRODUCTION"):
#     client = mlflow.tracking.MlflowClient()
#     client.transition_model_version_stage(
#         name=MODEL_REGISTRY_NAME,
#         version=int(version),
#         stage="Production"
#     )
#     st.success(f"🚀 Model v{version} promoted to **PRODUCTION**")

# # 7️⃣ MLflow UI
# st.subheader("8️⃣ MLflow UI")
# st.write("👉 Open in browser: http://127.0.0.1:5000")
# if st.button("Launch MLflow UI"):
#     mlflow_ui_command = [
#         "mlflow", "ui",
#         "--backend-store-uri", mlflow_db_uri,
#         "--default-artifact-root", mlflow_artifact_root,
#         "--host", "127.0.0.1",
#         "--port", "5000"
#     ]
#     subprocess.Popen(mlflow_ui_command)
#     st.success("🎉 MLflow UI launched!")
