# import streamlit as st
# from data_loader import load_data
# from model_registry import load_production_model_from_registry, get_model_type_from_registry
# from dashboard import run_dashboard
# from forecast_ui import run_forecast_app
# from training import setup_mlflow_training
# from monitoring import run_monitoring_app
# import os

# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================
# st.set_page_config(
#     layout="wide", 
#     page_title="Data Analysis & Forecast App", 
#     page_icon="📊"
# )

# # Constants
# DATA_PATH = 'Data.zip'

# # MLflow Configuration - USE RELATIVE PATHS
# mlflow_db_uri = "sqlite:///mlflow.db"

# # MLflow Setup with relative paths
# import mlflow

# # Set relative paths for MLflow
# mlflow.set_tracking_uri(mlflow_db_uri)

# # Create necessary directories if they don't exist
# os.makedirs("./mlruns", exist_ok=True)
# os.makedirs("./artifacts", exist_ok=True)

# # ============================================================================
# # MAIN ENTRY POINT
# # ============================================================================

# if __name__ == '__main__':
#     # Initialize Session State
#     defaults = {
#         'forecast_data': None,
#         'model_type': None,
#         'real_time_predictions': [],
#         'forecast_end_date': None,
#         'forecast_periods': None
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

#     # Load resources
#     train, min_date, max_date, sort_state, prophet_df = load_data()
    
#     # Load production model from MLflow Registry
#     model = load_production_model_from_registry()
#     model_type = get_model_type_from_registry() if model else "unknown"

#     # Sidebar Nav - UPDATED WITH MONITORING OPTION
#     st.sidebar.title("Navigation")
#     app_mode = st.sidebar.selectbox("Go to", 
#         ["Dashboard", "Forecast Engine", "Monitoring", "MLflow Tracking"]
#     )

#     if app_mode == "Dashboard":
#         run_dashboard(train, min_date, max_date, sort_state)
#     elif app_mode == "Forecast Engine":
#         run_forecast_app(model, prophet_df, model_type)
#     elif app_mode == "Monitoring":
#         run_monitoring_app()
#     else:  # MLflow Tracking
#         setup_mlflow_training()
# # [file content end]



# import streamlit as st
# from data_loader import load_data
# from model_registry import load_production_model_from_registry, get_model_type_from_registry
# from dashboard import run_dashboard
# from forecast_ui import run_forecast_app
# from training import setup_mlflow_training
# from monitoring import run_monitoring_app
# import os
# import mlflow
# import subprocess
# import tempfile

# # ============================================================================
# # AUTO-SETUP FUNCTIONS
# # ============================================================================

# def check_setup_completed():
#     """Check if setup has already been completed"""
#     required_paths = [
#         "./mlruns",
#         "./artifacts/prophet",
#         "./artifacts/arima", 
#         "./artifacts/lightgbm",
#         "./artifacts/best_models",
#         "./artifacts/prediction_logs",
#         "mlflow.db"
#     ]
    
#     # Check if all required directories and files exist
#     for path in required_paths:
#         if not os.path.exists(path):
#             return False
#     return True

# def setup_project_directories():
#     """Create all necessary directories"""
#     directories = [
#         "./mlruns",
#         "./artifacts",
#         "./artifacts/prophet", 
#         "./artifacts/arima",
#         "./artifacts/lightgbm",
#         "./artifacts/best_models",
#         "./artifacts/prediction_logs",
#         "./mlflow_project/models/models"
#     ]
    
#     for directory in directories:
#         os.makedirs(directory, exist_ok=True)
    
#     # Create MLflow database file
#     open("mlflow.db", "a").close()

# def setup_mlflow_experiments():
#     """Setup MLflow experiments and configuration"""
#     # Set MLflow configuration
#     mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
#     # Create experiments
#     experiments = {
#         "prophet_test": "./artifacts/prophet",
#         "arima_test": "./artifacts/arima", 
#         "lightgbm_test": "./artifacts/lightgbm",
#         "best_models": "./artifacts/best_models",
#         "prediction_logs": "./artifacts/prediction_logs"
#     }
    
#     for exp_name, artifact_location in experiments.items():
#         try:
#             # Create experiment if it doesn't exist
#             mlflow.create_experiment(
#                 name=exp_name,
#                 artifact_location=artifact_location
#             )
#         except Exception:
#             # Experiment already exists, which is fine
#             pass

# def setup_dvc_connection():
#     """Setup DVC connection if not configured"""
#     try:
#         # Check if DVC remote is already configured
#         result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
#         if result.returncode == 0 and result.stdout.strip():
#             return True  # DVC already configured
        
#         # Setup DVC remote
#         token = "d96854ae83eace50a396943c30be4aa5febf8eb3"
#         repo = "Mostafa2074/my-first-repo"
#         remote_name = "myremote"
        
#         commands = [
#             ["dvc", "init", "--subdir"],
#             ["dvc", "remote", "add", "-d", remote_name, f"https://{token}@github.com/{repo}.git"],
#             ["dvc", "remote", "modify", remote_name, "--local", "auth", "basic"],
#             ["dvc", "remote", "modify", remote_name, "--local", "user", token],
#         ]
        
#         for cmd in commands:
#             subprocess.run(cmd, capture_output=True)  # Silent execution
        
#         return True
#     except Exception:
#         return False  # DVC not available or setup failed

# def run_auto_setup():
#     """Run complete setup automatically"""
#     if check_setup_completed():
#         return True  # Setup already completed
    
#     st.sidebar.info("🚀 Running initial setup...")
    
#     try:
#         # Step 1: Create directories
#         with st.sidebar.status("Setting up project structure..."):
#             setup_project_directories()
#             st.sidebar.success("✅ Directories created")
        
#         # Step 2: Setup MLflow
#         with st.sidebar.status("Configuring MLflow..."):
#             setup_mlflow_experiments()
#             st.sidebar.success("✅ MLflow configured")
        
#         # Step 3: Try DVC setup (optional)
#         with st.sidebar.status("Checking DVC..."):
#             dvc_success = setup_dvc_connection()
#             if dvc_success:
#                 st.sidebar.success("✅ DVC configured")
#             else:
#                 st.sidebar.warning("⚠️ DVC setup skipped (not installed)")
        
#         st.sidebar.success("🎉 Setup completed successfully!")
#         return True
        
#     except Exception as e:
#         st.sidebar.error(f"❌ Setup failed: {e}")
#         return False

# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================
# st.set_page_config(
#     layout="wide", 
#     page_title="Data Analysis & Forecast App", 
#     page_icon="📊"
# )

# # Constants
# DATA_PATH = 'Data.zip'

# # ============================================================================
# # MAIN ENTRY POINT
# # ============================================================================

# if __name__ == '__main__':
#     # Run auto-setup on startup
#     if not run_auto_setup():
#         st.error("Project setup failed. Please check the errors above.")
#         st.stop()
    
#     # Initialize Session State
#     defaults = {
#         'forecast_data': None,
#         'model_type': None,
#         'real_time_predictions': [],
#         'forecast_end_date': None,
#         'forecast_periods': None
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

#     # Load resources
#     train, min_date, max_date, sort_state, prophet_df = load_data()
    
#     # Load production model from MLflow Registry
#     model = load_production_model_from_registry()
#     model_type = get_model_type_from_registry() if model else "unknown"

#     # Sidebar Nav - UPDATED WITH MONITORING OPTION
#     st.sidebar.title("Navigation")
#     st.sidebar.markdown("---")
#     app_mode = st.sidebar.selectbox("Go to", 
#         ["Dashboard", "Forecast Engine", "Monitoring", "MLflow Tracking"]
#     )

#     if app_mode == "Dashboard":
#         run_dashboard(train, min_date, max_date, sort_state)
#     elif app_mode == "Forecast Engine":
#         run_forecast_app(model, prophet_df, model_type)
#     elif app_mode == "Monitoring":
#         run_monitoring_app()
#     else:  # MLflow Tracking
#         setup_mlflow_training()












import streamlit as st
from data_loader import load_data
from model_registry import load_production_model_from_registry, get_model_type_from_registry
from dashboard import run_dashboard
from forecast_ui import run_forecast_app
from training import setup_mlflow_training
from monitoring import run_monitoring_app
import os
import mlflow
from mlflow.tracking import MlflowClient
import subprocess

# ============================================================================
# AUTO-SETUP FUNCTIONS
# ============================================================================

def check_setup_completed():
    """Check if setup has already been completed"""
    required_paths = [
        "mlflow.db",
        "./mlruns/0/meta.yaml"  # Check if default experiment exists
    ]
    
    # Check if all required directories and files exist
    for path in required_paths:
        if not os.path.exists(path):
            return False
    return True

def setup_project_directories():
    """Create all necessary directories"""
    directories = [
        "./mlruns",
        "./artifacts",
        "./artifacts/prophet", 
        "./artifacts/arima",
        "./artifacts/lightgbm",
        "./artifacts/best_models",
        "./artifacts/prediction_logs",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create MLflow database file
    open("mlflow.db", "a").close()

def setup_mlflow_experiments():
    """Setup MLflow experiments and configuration with proper error handling"""
    try:
        # Set MLflow configuration - USE RELATIVE PATHS ONLY
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Initialize MLflow client
        client = MlflowClient()
        
        # Create experiments with simple approach
        experiments = [
            "prophet_test",
            "arima_test", 
            "lightgbm_test",
            "best_models",
            "prediction_logs"
        ]
        
        for exp_name in experiments:
            try:
                # Check if experiment already exists
                experiment = client.get_experiment_by_name(exp_name)
                if experiment is None:
                    # Create new experiment with simple artifact location
                    mlflow.create_experiment(exp_name)
                    print(f"✅ Created experiment: {exp_name}")
            except Exception as e:
                print(f"⚠️  Could not create experiment {exp_name}: {e}")
                # Continue with other experiments
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow setup failed: {e}")
        return False

def setup_dvc_connection():
    """Setup DVC connection if not configured - SILENT OPERATION"""
    try:
        # Check if DVC is installed
        result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            return False  # DVC not installed
        
        # Check if DVC remote is already configured
        result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return True  # DVC already configured
        
        # Setup DVC remote silently
        token = "d96854ae83eace50a396943c30be4aa5febf8eb3"
        repo = "Mostafa2074/my-first-repo"
        remote_name = "myremote"
        
        commands = [
            ["dvc", "init", "--subdir", "-f"],
            ["dvc", "remote", "add", "-d", remote_name, f"https://{token}@github.com/{repo}.git"],
            ["dvc", "remote", "modify", remote_name, "--local", "auth", "basic"],
            ["dvc", "remote", "modify", remote_name, "--local", "user", token],
        ]
        
        for cmd in commands:
            subprocess.run(cmd, capture_output=True)  # Silent execution
        
        return True
    except Exception:
        return False  # DVC not available or setup failed

def run_auto_setup():
    """Run complete setup automatically"""
    if check_setup_completed():
        return True  # Setup already completed
    
    # Use try/except instead of st.sidebar.status to avoid conflicts
    try:
        # Step 1: Create directories
        setup_project_directories()
        
        # Step 2: Setup MLflow
        mlflow_success = setup_mlflow_experiments()
        if not mlflow_success:
            st.error("MLflow setup failed. Some features may not work.")
        
        # Step 3: Try DVC setup (optional)
        dvc_success = setup_dvc_connection()
        
        return True
        
    except Exception as e:
        st.error(f"Setup failed: {e}")
        return False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    layout="wide", 
    page_title="Data Analysis & Forecast App", 
    page_icon="📊"
)

# Constants
DATA_PATH = 'Data.zip'

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Run auto-setup on startup (minimal UI during setup)
    with st.spinner("Initializing application..."):
        setup_success = run_auto_setup()
    
    if not setup_success:
        st.error("Project setup failed. Please check the console for errors.")
        st.stop()
    
    # Initialize Session State
    defaults = {
        'forecast_data': None,
        'model_type': None,
        'real_time_predictions': [],
        'forecast_end_date': None,
        'forecast_periods': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Load resources
    train, min_date, max_date, sort_state, prophet_df = load_data()
    
    # Load production model from MLflow Registry with graceful failure handling
    try:
        model = load_production_model_from_registry()
        model_type = get_model_type_from_registry() if model else "unknown"
    except Exception as e:
        st.sidebar.warning("No production model found. Please train models in MLflow Tracking first.")
        model = None
        model_type = "unknown"

    # Sidebar Nav
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox("Go to", 
        ["Dashboard", "Forecast Engine", "Monitoring", "MLflow Tracking"]
    )

    if app_mode == "Dashboard":
        run_dashboard(train, min_date, max_date, sort_state)
    elif app_mode == "Forecast Engine":
        run_forecast_app(model, prophet_df, model_type)
    elif app_mode == "Monitoring":
        run_monitoring_app()
    else:  # MLflow Tracking
        setup_mlflow_training()



