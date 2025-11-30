import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import mlflow
import mlflow.pyfunc  # ✅ Only pyfunc needed
from mlflow.models.signature import infer_signature
import pickle
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

st.set_page_config(page_title="Multi-Model Forecast Trainer", layout="wide")
st.title("📊 Multi-Model Forecast Trainer with MLflow, DVC & Model Registry")

# Paths and DVC
dataset_dvc = r"model_dataset.csv.dvc"
prophet_model_dvc = r"mlflow_project/models/models.dvc"
arima_model_dvc = r"mlflow_project/models/models.dvc"
lgb_model_dvc = r"mlflow_project/models/models.dvc"

dataset_path = os.path.splitext(dataset_dvc)[0]
prophet_model_path = r"mlflow_project/models/models/prophet_tuned_model.pkl"
arima_model_path = r"mlflow_project/models/models/arima_model.pkl"
lgb_model_path = r"mlflow_project/models/models/Light GBM.pkl"

mlflow_db_uri = "sqlite:///mlflow.db"
mlflow_artifact_root = "file:///mlruns"

# Model Registry Names - CHANGED TO BestForecastModels
PROPHET_REGISTRY_NAME = "BestForecastModels"
ARIMA_REGISTRY_NAME = "BestForecastModels"
LIGHTGBM_REGISTRY_NAME = "BestForecastModels"
BEST_MODELS_EXPERIMENT = "best_models"

# MLflow Setup
mlflow.set_tracking_uri(mlflow_db_uri)

# ARIMA Model Wrapper for pyfunc
class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        # Handle both DataFrame and Series input
        if hasattr(model_input, 'index'):
            start_idx = model_input.index[0]
            end_idx = model_input.index[-1]
            return self.model.predict(start=start_idx, end=end_idx)
        else:
            # Fallback for other input types
            return self.model.predict(start=0, end=len(model_input)-1)

# Prophet Model Wrapper for pyfunc
class ProphetModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        # model_input should be a DataFrame with 'ds' column
        return self.model.predict(model_input)[['yhat']]

# LightGBM Model Wrapper for pyfunc  
class LightGBMModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        # model_input should be a DataFrame with feature columns
        return pd.DataFrame({'prediction': self.model.predict(model_input)})

# 1️⃣ Pull dataset and models from DVC
st.subheader("1️⃣ Pull dataset and pre-trained models from DVC")
if st.button("Pull from DVC"):
    try:
        subprocess.run(["dvc", "pull", dataset_dvc], check=True)
        subprocess.run(["dvc", "pull", prophet_model_dvc], check=True)
        subprocess.run(["dvc", "pull", arima_model_dvc], check=True)
        subprocess.run(["dvc", "pull", lgb_model_dvc], check=True)
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
    prophet_input_example = prophet_df[['ds']].head(5)  # Only need 'ds' for input
    
    # Prepare data for ARIMA
    series = df.set_index("date")["sales"]
    arima_input_example = series.head(10).to_frame()
    
    # Feature Engineering for LightGBM
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    # Prepare features and target for LightGBM
    feature_columns = ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 'weekofyear']
    X = df[feature_columns]
    y = df['sales']
    lgb_input_example = X.head(5)
    
    # Train/Test Split for Prophet
    prophet_train_df, prophet_test_df = train_test_split(prophet_df, test_size=0.2, shuffle=False)
    
    # Train/Test Split for ARIMA
    train_size = int(len(series) * 0.8)
    arima_train_series = series[:train_size]
    arima_test_series = series[train_size:]
    
    # Train/Test Split for LightGBM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    st.write(f"Training rows: {len(prophet_train_df)}, Testing rows: {len(prophet_test_df)}")
    
    # Store variables in session state to avoid re-computation
    st.session_state.df = df
    st.session_state.prophet_df = prophet_df
    st.session_state.prophet_train_df = prophet_train_df
    st.session_state.prophet_test_df = prophet_test_df
    st.session_state.series = series
    st.session_state.arima_train_series = arima_train_series
    st.session_state.arima_test_series = arima_test_series
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
else:
    st.error(f"Dataset not found: {dataset_path}")
    st.stop()

# 3️⃣ Load Pre-trained Models (Optional)
st.subheader("3️⃣ Load Pre-trained Models (Optional)")
col1, col2, col3 = st.columns(3)

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

with col3:
    lgb_pre_trained = None
    if os.path.exists(lgb_model_path):
        with open(lgb_model_path, "rb") as f:
            lgb_pre_trained = pickle.load(f)
        st.success("✅ Pre-trained LightGBM model loaded")

# 4️⃣ Prophet Model Training
st.subheader("4️⃣ Prophet Model Training")
prophet_param_grid = {
    "changepoint_prior_scale": [0.001],
    "seasonality_prior_scale": [0.1],
    "holidays_prior_scale": [0.1],
    "seasonality_mode": ["additive"],
    "yearly_seasonality": [True, False],
    "weekly_seasonality": [True, False]
}

if st.button("Train Prophet Model"):
    if 'prophet_train_df' not in st.session_state:
        st.error("Please load dataset first")
        st.stop()
        
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
                model.fit(st.session_state.prophet_train_df)

                # Forecast
                forecast_train = model.predict(st.session_state.prophet_train_df)
                y_train_pred = forecast_train["yhat"]
                train_rmse = mean_squared_error(st.session_state.prophet_train_df["y"], y_train_pred) ** 0.5
                train_mae = mean_absolute_error(st.session_state.prophet_train_df["y"], y_train_pred)

                forecast_test = model.predict(st.session_state.prophet_test_df)
                y_test_pred = forecast_test["yhat"]
                test_rmse = mean_squared_error(st.session_state.prophet_test_df["y"], y_test_pred) ** 0.5
                test_mae = mean_absolute_error(st.session_state.prophet_test_df["y"], y_test_pred)

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
            
            # ✅ Save as PYFUNC only
            wrapper_model = ProphetModelWrapper(best_prophet_model)
            
            # Generate signature
            prediction_output = best_prophet_model.predict(prophet_input_example)[['yhat']]
            signature = infer_signature(prophet_input_example, prediction_output)

            mlflow.pyfunc.log_model(
                python_model=wrapper_model,
                artifact_path="prophet_model",
                registered_model_name=PROPHET_REGISTRY_NAME,  # Now BestForecastModels
                input_example=prophet_input_example,
                signature=signature
            )

        st.success(f"🏆 Best Prophet Model registered to MLflow as PYFUNC (RMSE: {best_prophet_rmse:.2f})")

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
    if 'arima_train_series' not in st.session_state:
        st.error("Please load dataset first")
        st.stop()
        
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

                model = ARIMA(st.session_state.arima_train_series, order=(p, d, q))
                model_fit = model.fit()

                y_train_pred = model_fit.predict(start=st.session_state.arima_train_series.index[0], end=st.session_state.arima_train_series.index[-1])
                y_test_pred = model_fit.predict(start=st.session_state.arima_test_series.index[0], end=st.session_state.arima_test_series.index[-1])

                train_rmse = mean_squared_error(st.session_state.arima_train_series, y_train_pred) ** 0.5
                train_mae = mean_absolute_error(st.session_state.arima_train_series, y_train_pred)
                test_rmse = mean_squared_error(st.session_state.arima_test_series, y_test_pred) ** 0.5
                test_mae = mean_absolute_error(st.session_state.arima_test_series, y_test_pred)

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

        # ✅ Save as PYFUNC only
        wrapper_model = ARIMAModelWrapper(best_arima_model)
        
        # Generate signature
        prediction_output = best_arima_model.predict(
            start=arima_input_example.index[0], 
            end=arima_input_example.index[-1]
        )
        signature = infer_signature(arima_input_example, prediction_output)

        # Register to best_models experiment
        mlflow.set_experiment(BEST_MODELS_EXPERIMENT)
        with mlflow.start_run(run_name="Best_ARIMA_Model"):
            mlflow.log_params(best_arima_params)
            mlflow.log_metric("rmse", best_arima_rmse)
            mlflow.log_metric("mae", best_arima_mae)
            mlflow.set_tag("model_type", "arima")
            
            mlflow.pyfunc.log_model(
                python_model=wrapper_model,
                artifact_path="arima_model",
                registered_model_name=ARIMA_REGISTRY_NAME,  # Now BestForecastModels
                signature=signature,
                input_example=arima_input_example
            )

        st.success(f"🏆 Best ARIMA model registered to MLflow as PYFUNC (RMSE: {best_arima_rmse:.2f})")

        try:
            subprocess.run(["dvc", "add", arima_model_path], check=True)
            subprocess.run(["dvc", "push"], check=True)
            st.success("📦 ARIMA model updated in DVC!")
        except Exception as e:
            st.warning(f"ARIMA DVC Push failed: {e}")

# 6️⃣ LightGBM Model Training
st.subheader("6️⃣ LightGBM Model Training (RandomizedSearchCV)")

# Parameter grid for RandomizedSearch
lgb_param_grid = {
    'num_leaves': [31],
    'max_depth': [-1],
    'learning_rate': [0.005],
    'n_estimators': [500],
    'feature_fraction': [0.7],
    'bagging_fraction': [0.7],
    'bagging_freq': [1],
    'lambda_l1': [1],
    'lambda_l2': [1],
    'min_data_in_leaf': [80]
}

if st.button("Train LightGBM Model"):
    if 'X_train' not in st.session_state:
        st.error("Please load dataset first")
        st.stop()
        
    st.write("🚀 Starting LightGBM RandomizedSearchCV...")
    
    # Set up LightGBM experiment
    mlflow.set_experiment("lightgbm_test")
    
    best_lgb_rmse = float("inf")
    best_lgb_mae = float("inf")
    best_lgb_params = None
    best_lgb_model = None

    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Initialize LightGBM regressor
    lgb_reg = lgb.LGBMRegressor(
        random_state=42,
        verbose=-1,
        force_row_wise=True
    )

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgb_reg,
        param_distributions=lgb_param_grid,
        n_iter=25,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    st.text("⏳ Running LightGBM Hyperparameter Tuning...")
    
    # Start MLflow run for the entire tuning process
    with mlflow.start_run(run_name="LightGBM_RandomizedSearch"):
        # Fit the randomized search
        random_search.fit(st.session_state.X_train, st.session_state.y_train)
        
        # Update progress
        progress_bar.progress(1.0)
        progress_text.text("✅ Tuning completed!")
        
        # Get best model and parameters
        best_lgb_model = random_search.best_estimator_
        best_lgb_params = random_search.best_params_
        best_score = -random_search.best_score_

        # Make predictions
        y_train_pred = best_lgb_model.predict(st.session_state.X_train)
        y_test_pred = best_lgb_model.predict(st.session_state.X_test)

        # Calculate metrics
        train_rmse = mean_squared_error(st.session_state.y_train, y_train_pred) ** 0.5
        train_mae = mean_absolute_error(st.session_state.y_train, y_train_pred)
        test_rmse = mean_squared_error(st.session_state.y_test, y_test_pred) ** 0.5
        test_mae = mean_absolute_error(st.session_state.y_test, y_test_pred)

        # Log parameters and metrics
        mlflow.log_params(best_lgb_params)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("best_cv_mae", best_score)

    st.success(f"🎉 LightGBM RandomizedSearchCV complete. Best RMSE={test_rmse:.2f}, Best MAE={test_mae:.2f}")
    st.write("Best LightGBM Parameters:", best_lgb_params)

    # Save and register best LightGBM model
    if best_lgb_model is not None:
        # Save locally
        os.makedirs(os.path.dirname(lgb_model_path), exist_ok=True)
        with open(lgb_model_path, "wb") as f:
            pickle.dump(best_lgb_model, f)

        # Register to best_models experiment
        mlflow.set_experiment(BEST_MODELS_EXPERIMENT)
        with mlflow.start_run(run_name="Best_LightGBM_Model"):
            mlflow.log_params(best_lgb_params)
            mlflow.log_metric("rmse", test_rmse)
            mlflow.log_metric("mae", test_mae)
            mlflow.set_tag("model_type", "lightgbm")
            
            # ✅ Save as PYFUNC only
            wrapper_model = LightGBMModelWrapper(best_lgb_model)
            
            # Generate signature
            prediction_output = best_lgb_model.predict(lgb_input_example)
            signature = infer_signature(lgb_input_example, pd.DataFrame({'prediction': prediction_output}))

            mlflow.pyfunc.log_model(
                python_model=wrapper_model,
                artifact_path="lightgbm_model",
                registered_model_name=LIGHTGBM_REGISTRY_NAME,  # Now BestForecastModels
                input_example=lgb_input_example,
                signature=signature
            )

        st.success(f"🏆 Best LightGBM Model registered to MLflow as PYFUNC (RMSE: {test_rmse:.2f})")

        # DVC update
        try:
            subprocess.run(["dvc", "add", lgb_model_path], check=True)
            subprocess.run(["dvc", "push"], check=True)
            st.success("📦 LightGBM model updated in DVC!")
        except Exception as e:
            st.warning(f"LightGBM DVC Push failed: {e}")

# 7️⃣ Model Registry Management - Consolidated for BestForecastModels
st.subheader("7️⃣ Model Registry Management")
st.write("Promote models in **BestForecastModels** registry:")

col1, col2 = st.columns(2)

with col1:
    model_version = st.text_input("Model version to promote:", "1")
    
with col2:
    target_stage = st.selectbox("Target stage:", ["Staging", "Production", "Archived"])

if st.button(f"Promote BestForecastModels v{model_version} to {target_stage.upper()}"):
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="BestForecastModels", 
            version=int(model_version), 
            stage=target_stage
        )
        st.success(f"✅ BestForecastModels v{model_version} promoted to **{target_stage.upper()}**")
    except Exception as e:
        st.error(f"Error promoting model: {e}")

# 8️⃣ MLflow UI
st.subheader("8️⃣ MLflow UI")
st.write("👉 Open in browser: http://127.0.0.1:5000")
if st.button("Launch MLflow UI"):
    try:
        mlflow_ui_command = [
            "mlflow", "ui",
            "--backend-store-uri", mlflow_db_uri,
            "--default-artifact-root", mlflow_artifact_root,
            "--host", "127.0.0.1",
            "--port", "5000"
        ]
        subprocess.Popen(mlflow_ui_command)
        st.success("🎉 MLflow UI launched!")
    except Exception as e:
        st.error(f"Error launching MLflow UI: {e}")

# 9️⃣ Experiment Status
st.subheader("9️⃣ Experiment Status")
if st.button("Check Best Models Experiment"):
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(BEST_MODELS_EXPERIMENT)
        if experiment:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            st.write(f"**Best Models Experiment:** {BEST_MODELS_EXPERIMENT}")
            st.write(f"**Total Best Model Runs:** {len(runs)}")
            
            if runs:
                st.write("**Registered Best Models (ALL as PYFUNC):**")
                for run in runs:
                    model_type = run.data.tags.get("model_type", "unknown")
                    rmse = run.data.metrics.get("rmse", "N/A")
                    st.write(f"- {run.info.run_name} ({model_type}) - RMSE: {rmse}")
        else:
            st.info("Best Models experiment not created yet. Train models first.")
    except Exception as e:
        st.error(f"Error checking experiment status: {e}")

# streamlit run "D:\Final Project\mlflow_project\train_models_streamlit.py"