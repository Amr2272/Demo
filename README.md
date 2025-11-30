# Deploying to Streamlit Cloud

This repository contains a Streamlit-based forecasting app. The primary Streamlit entrypoint is `main.py`.

Quick deploy steps:

1. Push this repository to GitHub.
2. On Streamlit Cloud (https://share.streamlit.io), click **New app** → select this GitHub repo → set the **Main file** to `main.py` and the branch to deploy.
3. Streamlit Cloud will install dependencies from `requirements.txt` and start the app with `streamlit run main.py`.

Notes & recommendations
- **Entry point**: use `main.py` (it provides navigation to the Dashboard, Forecast UI and Monitoring pages).
- **Requirements**: `requirements.txt` includes heavy packages such as `prophet`, `lightgbm` and `mlflow`. These may need build tools or fail to install on the Cloud if wheels are not available. If you hit install errors:
  - Try pinning to a version that has pre-built wheels for Linux; or
  - Remove the package from `requirements.txt` and serve pre-built model artifacts instead.
- **MLflow**: The app uses MLflow for models and tracking. Using a local SQLite `mlflow.db` inside Streamlit Cloud is ephemeral and won't persist across deploys. Recommended options:
  - Use an external MLflow tracking server (set `MLFLOW_TRACKING_URI` via Streamlit Cloud Secrets), or
  - Store production model artifacts directly in the repo under `artifacts/` (small models) or use cloud storage (S3) and configure MLflow accordingly.
- **DVC**: DVC is used in development. Streamlit Cloud does not automatically fetch DVC tracked large files; use `artifacts/` in repo or host models externally.

Setting secrets in Streamlit Cloud
- In the app deploy settings, under **Secrets**, add keys such as `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT`, or any credentials required to access remote model storage.

Local testing
- Create a conda/venv environment and install dependencies:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
streamlit run main.py
```

If you'd like, I can:
- create a pinned `requirements.txt` with tested versions,
- generate a small `streamlit_app.py` that selects `main.py` automatically, or
- help push this repo to GitHub and connect Streamlit Cloud.
