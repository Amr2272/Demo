import streamlit as st
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

@st.cache_resource
def load_production_model_from_registry(model_name="BestForecastModels", stage="Production"):
    """Loads the production model from MLflow Model Registry."""
    try:
        client = MlflowClient()
        
        # Get the latest production model version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        production_models = [mv for mv in model_versions if mv.current_stage == stage]
        
        if not production_models:
            st.warning(f"No production model found in registry for {model_name}. Checking for any staged model...")
            # Fallback to any model version
            if model_versions:
                latest_version = max(model_versions, key=lambda x: int(x.version))
                model_uri = f"models:/{model_name}/{latest_version.version}"
                st.info(f"Using latest version {latest_version.version} (stage: {latest_version.current_stage})")
            else:
                st.error(f"No models found in registry for {model_name}")
                return None
        else:
            # Get the latest production model
            latest_production = max(production_models, key=lambda x: int(x.version))
            model_uri = f"models:/{model_name}/{latest_production.version}"
            st.success(f"Loaded production model: {model_name} version {latest_production.version}")
        
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)
        return model
        
    except Exception as e:
        st.error(f"Error loading model from registry: {e}")
        return None

def get_model_type_from_registry(model_name="BestForecastModels", stage="Production"):
    """Determine the type of model in the registry."""
    try:
        client = MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        staged_models = [mv for mv in model_versions if mv.current_stage == stage]
        
        if staged_models:
            latest_model = max(staged_models, key=lambda x: int(x.version))
            run_id = latest_model.run_id
            
            # Get the run details to check model type
            run = client.get_run(run_id)
            model_type = run.data.tags.get("model_type", "unknown")
            return model_type
        return "unknown"
    except Exception as e:
        st.error(f"Error determining model type: {e}")
        return "unknown"



