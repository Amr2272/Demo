# import subprocess
# import pandas as pd

# # 1️⃣ Ensure the dataset is pulled from remote
# subprocess.run(["dvc", "pull", "model_dataset.csv.dvc"], check=True)

# # 2️⃣ Restore the file to the workspace
# subprocess.run(["dvc", "checkout"], check=True)

# # 3️⃣ Now read it with pandas
# data = pd.read_csv("model_dataset.csv")  # or data/model_dataset.csv if you want it in a folder
# print(data.head())


from mlflow.tracking import MlflowClient

MODEL_NAME = "ProphetForecastModel"
client = MlflowClient()

# List all versions
versions = client.get_latest_versions(MODEL_NAME)

# Delete each version
for v in versions:
    print(f"Deleting version {v.version}")
    client.delete_model_version(name=MODEL_NAME, version=v.version)

# Optionally delete the registered model itself
# client.delete_registered_model(MODEL_NAME)
