import subprocess
import os
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.prophet
import pickle
from itertools import product

# -----------------------------
# 0️⃣ DVC integration
# -----------------------------
dataset_dvc = r"D:\Final Project\mlflow_project\data\model_dataset.csv.dvc"
prophet_model_dvc = r"D:\Final Project\mlflow_project\models\models.dvc"

# Pull dataset from DVC
print(f"Pulling dataset from DVC: {dataset_dvc}")
subprocess.run(["dvc", "pull", dataset_dvc], check=True)
subprocess.run(["dvc", "checkout"], check=True)

# Pull pre-trained Prophet model from DVC
print(f"Pulling Prophet model from DVC: {prophet_model_dvc}")
subprocess.run(["dvc", "pull", prophet_model_dvc], check=True)
subprocess.run(["dvc", "checkout"], check=True)

# Local paths (strings only)
dataset_path = os.path.splitext(dataset_dvc)[0]
prophet_model_path = r"D:\Final Project\mlflow_project\models\models\prophet_tuned_model.pkl"

print(f"✅ Dataset path: {dataset_path}")
print(f"✅ Prophet model path: {prophet_model_path}")

# -----------------------------
# 1️⃣ MLflow setup (SQLite backend)
# -----------------------------
mlflow_db_uri = "sqlite:///D:/Final_Project/mlflow_project/mlflow.db"
mlflow_artifact_root = r"D:\Final Project\mlflow_project\mlruns"

mlflow.set_tracking_uri(mlflow_db_uri)
experiment_name = "sales_forecast_prophet"
mlflow.set_experiment(experiment_name)

# -----------------------------
# 2️⃣ Load dataset
# -----------------------------
df = pd.read_csv(dataset_path, parse_dates=["date"])
prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
input_example = prophet_df.head(5)

# -----------------------------
# 3️⃣ Log pre-trained Prophet model (if exists)
# -----------------------------
if os.path.exists(prophet_model_path):
    with open(prophet_model_path, "rb") as f:
        pre_trained_model = pickle.load(f)

    with mlflow.start_run(run_name="PreTrained_Prophet_Model"):
        mlflow.prophet.log_model(
            pre_trained_model,
            name="prophet_model",
            input_example=input_example
        )
        print("✅ Pre-trained Prophet model logged to MLflow!")
else:
    print("⚠️ Pre-trained Prophet model not found, skipping.")

# -----------------------------
# 4️⃣ Train new Prophet model grid search
# -----------------------------
param_grid = {
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    "seasonality_prior_scale": [0.1, 1.0, 10.0],
    "holidays_prior_scale": [0.1, 1.0, 10.0],
    "seasonality_mode": ["additive", "multiplicative"],
    "yearly_seasonality": [True, False],
    "weekly_seasonality": [True, False]
}

best_rmse = float("inf")
best_params = None

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
            # Log hyperparameters
            mlflow.log_params(params)

            # Train Prophet model
            model = Prophet(**params)
            model.fit(prophet_df)

            # Forecast
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            y_true = prophet_df["y"]
            y_pred = forecast["yhat"][:len(y_true)]
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            mae = mean_absolute_error(y_true, y_pred)

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # Log trained model
            mlflow.prophet.log_model(model, name="prophet_model", input_example=input_example)

            # Track best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

            print(f"✅ Run {run_name} completed. RMSE: {rmse}, MAE: {mae}")

    except Exception as e:
        print(f"❌ Run {run_name} failed: {e}")

print("✅ Best RMSE:", best_rmse)
print("✅ Best hyperparameters:", best_params)

# -----------------------------
# 5️⃣ Save best model and update DVC
# -----------------------------
if best_params is not None:
    print("🚀 Saving best Prophet model locally and updating DVC...")
    best_model = Prophet(**best_params)
    best_model.fit(prophet_df)  # retrain on full dataset with best params

    # Save as pickle
    with open(prophet_model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"✅ Best model saved to {prophet_model_path}")

    # Enable auto-staging for DVC so git add is automatic
    subprocess.run(["dvc", "config", "core.autostage", "true"], check=True)

    # DVC add & push
    subprocess.run(["dvc", "add", prophet_model_path], check=True)
    subprocess.run(["dvc", "push"], check=True)
    print("✅ DVC model updated and pushed to remote!")
else:
    print("⚠️ No best model found, skipping DVC update.")

# -----------------------------
# 6️⃣ Launch MLflow UI programmatically
# -----------------------------
print("🚀 Launching MLflow UI...")
mlflow_ui_command = [
    "mlflow", "ui",
    "--backend-store-uri", mlflow_db_uri,
    "--default-artifact-root", mlflow_artifact_root,
    "--host", "127.0.0.1",
    "--port", "5000"
]

# Start MLflow UI as a background process
subprocess.Popen(mlflow_ui_command)
print("✅ MLflow UI should be running at http://127.0.0.1:5000")
