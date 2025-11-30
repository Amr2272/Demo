import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pickle
import lightgbm as lgb

st.set_page_config(page_title="LightGBM Forecast Trainer", layout="wide")
st.title("📊 LightGBM Forecast Trainer with MLflow, DVC & Model Registry")

# Paths and DVC
dataset_dvc = r"D:\Final Project\model_dataset.csv.dvc"
lgb_model_dvc = r"D:\Final Project\mlflow_project\models\models.dvc"
dataset_path = os.path.splitext(dataset_dvc)[0]
lgb_model_path = r"D:\Final Project\mlflow_project\models\models\Light GBM.pkl"

mlflow_db_uri = "sqlite:///D:/Final_Project/mlflow_project/mlflow.db"
mlflow_artifact_root = "file:///D:/Final_Project/mlflow_project/mlruns"

MODEL_REGISTRY_NAME = "LightgbmForecastModel"

# 1️⃣ Pull dataset and model from DVC
st.subheader("1️⃣ Pull dataset and pre-trained model from DVC")
if st.button("Pull from DVC"):
    st.text(f"Pulling dataset from DVC: {dataset_dvc}")
    try:
        subprocess.run(["dvc", "pull", dataset_dvc], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        
        st.text(f"Pulling LightGBM model from DVC: {lgb_model_dvc}")
        subprocess.run(["dvc", "pull", lgb_model_dvc], check=True)
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
experiment_name = "lightgbm_test"

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
    
    # Feature Engineering for LightGBM
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    # Prepare features and target
    feature_columns = ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 'weekofyear']
    X = df[feature_columns]
    y = df['sales']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    st.write(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
    
    # Create input example for MLflow signature
    input_example = X_train.head(5)
else:
    st.error(f"Dataset not found: {dataset_path}")
    st.stop()

# 4️⃣ Load Pre-trained Model Locally (Optional)
st.subheader("4️⃣ Load Pre-trained Model (Optional)")
pre_trained_model = None
if os.path.exists(lgb_model_path):
    with open(lgb_model_path, "rb") as f:
        pre_trained_model = pickle.load(f)
    st.success("✅ Pre-trained model loaded locally (not logged to MLflow)")

# 5️⃣ Train LightGBM Model (RandomizedSearchCV)
st.subheader("5️⃣ Train LightGBM Model (RandomizedSearchCV)")

# Parameter grid for RandomizedSearch
param_grid = {
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

if st.button("Start RandomizedSearchCV"):
    best_rmse = float("inf")
    best_mae = float("inf")
    best_params = None
    best_model_obj = None

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
        param_distributions=param_grid,
        n_iter=25,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    st.text("⏳ Running Hyperparameter Tuning...")
    
    # Start MLflow run for the entire tuning process
    with mlflow.start_run(run_name="LightGBM_RandomizedSearch"):
        # Fit the randomized search
        random_search.fit(X_train, y_train)
        
        # Update progress
        progress_bar.progress(1.0)
        progress_text.text("✅ Tuning completed!")
        
        # Get best model and parameters
        best_model_obj = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = -random_search.best_score_

        # Make predictions
        y_train_pred = best_model_obj.predict(X_train)
        y_test_pred = best_model_obj.predict(X_test)

        # Calculate metrics
        train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("best_cv_mae", best_score)

    st.success(f"🎉 RandomizedSearchCV complete. Best RMSE={test_rmse:.2f}, Best MAE={test_mae:.2f}")
    st.write("Best Parameters:", best_params)

    # 6️⃣ Register ONLY the Best Model
    if best_model_obj is not None:
        # Save locally
        os.makedirs(os.path.dirname(lgb_model_path), exist_ok=True)
        with open(lgb_model_path, "wb") as f:
            pickle.dump(best_model_obj, f)

        # Log to MLflow
        with mlflow.start_run(run_name="Best_Model_Final"):
            mlflow.log_params(best_params)
            mlflow.log_metric("rmse", test_rmse)
            mlflow.log_metric("mae", test_mae)
            
            # Generate Signature
            signature = infer_signature(input_example, best_model_obj.predict(input_example))

            mlflow.lightgbm.log_model(
                best_model_obj,
                artifact_path="lightgbm_model",
                registered_model_name=MODEL_REGISTRY_NAME,
                input_example=input_example,
                signature=signature
            )

        st.success(f"🏆 Only the Best Model (RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}) was registered to MLflow.")

        # DVC update
        try:
            subprocess.run(["dvc", "add", lgb_model_path], check=True)
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