# import pandas as pd
# from prophet import Prophet
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import mlflow
# import mlflow.prophet
# from itertools import product

# # === MLflow setup ===
# mlflow.set_tracking_uri("file:///D:/Final Project/mlflow_project/mlruns")
# mlflow.set_experiment("sales_forecast_prophet")

# # === Load dataset ===
# df = pd.read_csv(r"D:\Final Project\mlflow_project\data\model_dataset.csv", parse_dates=["date"])
# prophet_df = df.rename(columns={"date": "ds", "sales": "y"})

# # === Hyperparameter grid ===
# param_grid = {
#     "changepoint_prior_scale": [0.001],
#     "seasonality_prior_scale": [0.1],
#     "holidays_prior_scale": [0.1],
#     "seasonality_mode": ["additive"],
#     "yearly_seasonality": [True],
#     "weekly_seasonality": [True]
# }

# best_rmse = float("inf")
# best_params = None

# # Example input for MLflow logging
# input_example = prophet_df.head(5)

# # === Grid search ===
# for cps, sps, hps, mode, yearly, weekly in product(
#     param_grid["changepoint_prior_scale"],
#     param_grid["seasonality_prior_scale"],
#     param_grid["holidays_prior_scale"],
#     param_grid["seasonality_mode"],
#     param_grid["yearly_seasonality"],
#     param_grid["weekly_seasonality"]
# ):
#     with mlflow.start_run(nested=True):
#         params = {
#             "changepoint_prior_scale": cps,
#             "seasonality_prior_scale": sps,
#             "holidays_prior_scale": hps,
#             "seasonality_mode": mode,
#             "yearly_seasonality": yearly,
#             "weekly_seasonality": weekly,
#             "daily_seasonality": False
#         }
        
#         # Log hyperparameters
#         mlflow.log_params(params)
        
#         # Train Prophet
#         model = Prophet(**params)
#         model.fit(prophet_df)
        
#         # Forecast next 30 days
#         future = model.make_future_dataframe(periods=30)
#         forecast = model.predict(future)
        
#         # Compute metrics on original data
#         y_true = prophet_df["y"]
#         y_pred = forecast["yhat"][:len(y_true)]
#         rmse = mean_squared_error(y_true, y_pred) ** 0.5
#         mae = mean_absolute_error(y_true, y_pred)
        
#         # Log metrics
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("mae", mae)
        
#         # Track best model
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_params = params
        
#         # Log Prophet model
#         mlflow.prophet.log_model(
#             model, 
#             artifact_path="prophet_model",
#             input_example=input_example
#         )

# print(f"✅ Best RMSE: {best_rmse}")
# print(f"✅ Best hyperparameters: {best_params}")


from mlflow.tracking import MlflowClient

model_name = "ProphetForecastModel"
client = MlflowClient()

versions = client.search_model_versions(f"name='{model_name}'")

for v in versions:
    version = v.version
    print(f"Archiving and deleting version {version}...")

    # Archive version (required)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Archived"
    )

    # Now delete
    client.delete_model_version(
        name=model_name,
        version=version
    )

print("All versions deleted successfully!")
