import mlflow
from mlflow.tracking import MlflowClient
import yaml
import os

# Load model registry and Dagshub parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

repo_owner = params["mlflow"]["repo_owner"]
repo_name = params["mlflow"]["repo_name"]
model_name = params["model_registration"]["model_name"]

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


def test_fetch_latest_production_model():
    client = MlflowClient()

    # Fetch latest model in Production stage
    prod_models = client.get_latest_versions(name=model_name, stages=["Production"])

    assert prod_models, f"No model version found in 'Production' for model '{model_name}'"

    latest_model = prod_models[0]
    model_uri = f"models:/{model_name}/{latest_model.version}"

    print(f"Latest Production Model URI: {model_uri}")
    print(f"Version: {latest_model.version}, Run ID: {latest_model.run_id}")

    # Load the model to verify
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_uri}: {e}")


if __name__ == "__main__":
    test_fetch_latest_production_model()