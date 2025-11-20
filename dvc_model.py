import subprocess
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import joblib
import pickle
# from statsmodels.tsa.arima.model import ARIMAResults
from prophet import Prophet

# -----------------------------
# 1️⃣ Pull models via DVC
# -----------------------------
subprocess.run(["dvc", "pull", "models.dvc"], check=True)
subprocess.run(["dvc", "checkout"], check=True)

models_path = "models"

# -----------------------------
# 2️⃣ Define PythonModel wrappers
# -----------------------------
class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        # model_input should have 'ds' column for Prophet
        return self.model.predict(model_input)

# Optional: ARIMA wrapper if you want to log ARIMA
# class ARIMAWrapper(mlflow.pyfunc.PythonModel):
#     def __init__(self, model):
#         self.model = model
#     def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
#         # Example: forecast n steps equal to number of rows in input
#         return self.model.forecast(steps=len(model_input))

# -----------------------------
# 3️⃣ Set or create MLflow experiment
# -----------------------------
experiment_name = "MyModelsExperiment"
mlflow.set_experiment(experiment_name)
print("Logging to experiment:", experiment_name)

# -----------------------------
# 4️⃣ Start MLflow run and log models
# -----------------------------
with mlflow.start_run():

    # --- XGBoost model ---
    xgb_file = os.path.join(models_path, "xgboost_model.pkl")
    xgb_model = joblib.load(xgb_file)
    mlflow.sklearn.log_model(
        sk_model=xgb_model,
        artifact_path="xgb_model",
        registered_model_name="XGBoostModel"
    )
    print("XGBoost logged.")

    # --- ARIMA model (optional) ---
    # arima_file = os.path.join(models_path, "arima_model.pkl")
    # arima_model = ARIMAResults.load(arima_file)
    # mlflow.pyfunc.log_model(
    #     artifact_path="arima_model",
    #     python_model=ARIMAWrapper(arima_model),
    #     registered_model_name="ARIMAModel"
    # )
    # print("ARIMA logged.")

    # --- Prophet model ---
    prophet_file = os.path.join(models_path, "prophet_tuned_model.pkl")
    with open(prophet_file, "rb") as f:
        prophet_model = pickle.load(f)

    # Provide an input example for signature
    input_example = pd.DataFrame({"ds": ["2025-11-15"]})

    mlflow.pyfunc.log_model(
        artifact_path="prophet_model",
        python_model=ProphetWrapper(prophet_model),
        registered_model_name="ProphetModel",
        input_example=input_example
    )
    print("Prophet logged.")

print("All models logged successfully!")
