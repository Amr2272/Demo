import streamlit as st
from typing import Optional

# Lazily import heavy dependencies (mlflow, requests) inside functions so
# the app doesn't crash at import-time on Streamlit Cloud when builds fail.


@st.cache_resource
def load_production_model_from_registry(model_name="BestForecastModels", stage="Production"):
    import streamlit as st
    import mlflow
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient


    @st.cache_resource
    def load_production_model_from_registry(model_name="BestForecastModels", stage="Production"):
        """Loads the production model from MLflow Model Registry.

        Logic:
        - If `MODEL_ARTIFACT_URL` secret is provided, try downloading and loading it.
        - Otherwise, query the MLflow Model Registry, attempt to load `models:/...`.
        - If that fails, try downloading artifacts referenced by the model version.
        - Final fallback: search local repo `artifacts/` for a folder containing `MLmodel`.
        """

        # 1) Secret-based override: download model from URL if provided
        try:
            model_url = None
            if isinstance(st.secrets, dict):
                model_url = (
                    st.secrets.get("MODEL_ARTIFACT_URL")
                    or st.secrets.get("MODEL_URL")
                    or st.secrets.get("MODEL_ZIP_URL")
                )
            if model_url:
                try:
                    import tempfile
                    import os
                    import zipfile
                    import shutil

                    # requests may not be available at import time; import lazily
                    try:
                        import requests
                    except Exception as e:
                        st.error("Required package 'requests' is not installed in the deployment. Add 'requests' to requirements.txt and redeploy.")
                        return None

                    tmpdir = tempfile.mkdtemp(prefix="model_url_")
                    dest_path = os.path.join(tmpdir, "downloaded_model")

                    r = requests.get(model_url, stream=True)
                    r.raise_for_status()

                    if model_url.lower().endswith(".zip"):
                        zip_path = os.path.join(tmpdir, "model.zip")
                        with open(zip_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(dest_path)

                        candidate = None
                        for root, dirs, files in os.walk(dest_path):
                            if "MLmodel" in files:
                                candidate = root
                                break
                        if candidate is None:
                            candidate = dest_path
                    else:
                        os.makedirs(dest_path, exist_ok=True)
                        fname = os.path.join(dest_path, os.path.basename(model_url))
                        with open(fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        candidate = dest_path

                    # mlflow may not be installed in the cloud. Import lazily and handle errors.
                    try:
                        import mlflow
                        import mlflow.pyfunc
                    except Exception:
                        st.error("Required package 'mlflow' is not installed in the deployment. Add 'mlflow' to requirements.txt and redeploy.")
                        return None

                    try:
                        model = mlflow.pyfunc.load_model(candidate)
                        try:
                            shutil.rmtree(tmpdir)
                        except Exception:
                            pass
                        return model
                    except Exception as e:
                        st.warning(f"Model download from secret URL failed to load: {e}")
        except Exception:
            # Non-fatal: proceed to registry logic
            pass

        # 2) Registry-based loading
        try:
            try:
                from mlflow.tracking import MlflowClient
                import mlflow
                import mlflow.pyfunc
            except Exception:
                st.error("Required package 'mlflow' is not installed in the deployment. Add 'mlflow' to requirements.txt and redeploy.")
                return None

            client = MlflowClient()

            # Get the latest production model version
            model_versions = client.search_model_versions(f"name='{model_name}'")
            production_models = [mv for mv in model_versions if mv.current_stage == stage]

            if not production_models:
                st.warning(f"No production model found in registry for {model_name}. Checking for any staged model...")
                if model_versions:
                    latest_version = max(model_versions, key=lambda x: int(x.version))
                    model_uri = f"models:/{model_name}/{latest_version.version}"
                    st.info(f"Using latest version {latest_version.version} (stage: {latest_version.current_stage})")
                else:
                    st.error(f"No models found in registry for {model_name}")
                    return None
            else:
                latest_production = max(production_models, key=lambda x: int(x.version))
                model_uri = f"models:/{model_name}/{latest_production.version}"
                st.success(f"Loaded production model: {model_name} version {latest_production.version}")

            # Debug: show model version info in sidebar for troubleshooting
            try:
                expander = st.sidebar.expander("Model Registry Debug")
                with expander:
                    st.write("Model versions found:")
                    for mv in model_versions:
                        st.write({
                            "version": mv.version,
                            "stage": mv.current_stage,
                            "run_id": getattr(mv, "run_id", None),
                            "source": getattr(mv, "source", None),
                        })
            except Exception:
                pass

            # Try loading the model directly
            try:
                model = mlflow.pyfunc.load_model(model_uri)
                return model
            except Exception as load_err:
                st.warning(f"Direct load failed: {load_err}. Trying artifact download fallback...")

                try:
                    import tempfile
                    import shutil
                    import re
                    from mlflow import artifacts

                    source = None
                    try:
                        source = latest_production.source  # type: ignore
                    except Exception:
                        try:
                            source = latest_version.source  # type: ignore
                        except Exception:
                            if model_versions:
                                source = model_versions[-1].source

                    if not source:
                        raise RuntimeError("Could not determine model artifact source URI")

                    def _normalize_source_uri(src):
                        if isinstance(src, str):
                            if re.match(r"^/[A-Za-z]:/", src) or re.match(r"^/[A-Za-z]:\\", src):
                                return "file://" + src.replace("\\\\", "/")
                            if re.match(r"^[A-Za-z]:\\\\", src) or re.match(r"^[A-Za-z]:/", src):
                                return "file:///" + src.replace("\\\\", "/")
                        return src

                    normalized = _normalize_source_uri(source)

                    # Debug info in sidebar
                    try:
                        dbg = st.sidebar.expander("Model Load Debug Info")
                        with dbg:
                            st.write({
                                "selected_model_uri": model_uri,
                                "resolved_source": source,
                                "normalized_source": normalized,
                            })
                    except Exception:
                        pass

                    tmpdir = tempfile.mkdtemp(prefix="mlflow_model_")
                    local_path = artifacts.download_artifacts(artifact_uri=normalized, dst_path=tmpdir)

                    model = mlflow.pyfunc.load_model(local_path)

                    try:
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass

                    return model
                except Exception as fallback_err:
                    st.warning(f"Artifact download fallback failed: {fallback_err}")

                    # FINAL FALLBACK: try loading model from repository `artifacts/` folder.
                    try:
                        import os

                        repo_artifacts_root = os.path.join(os.getcwd(), "artifacts")
                        if os.path.exists(repo_artifacts_root):
                            for root, dirs, files in os.walk(repo_artifacts_root):
                                if "MLmodel" in files:
                                    try:
                                        st.info(f"Attempting to load model from repo artifacts: {root}")
                                        model = mlflow.pyfunc.load_model(root)
                                        return model
                                    except Exception:
                                        continue

                        st.error("Error loading model from registry (fallbacks failed). No usable artifacts found in repo `artifacts/`.")
                    except Exception as final_err:
                        st.error(f"Final fallback failed: {final_err}")
                    return None

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



