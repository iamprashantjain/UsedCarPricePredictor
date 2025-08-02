import asyncio
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel
import yaml
from fastapi.responses import JSONResponse

# Load config
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

repo_owner = params["mlflow"]["repo_owner"]
repo_name = params["mlflow"]["repo_name"]
model_name = params["model_registration"]["model_name"]

# Set Dagshub token
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

# Load model
model: mlflow.pyfunc.PyFuncModel = None

def load_production_model():
    client = MlflowClient()

    # Fetch latest model in Production stage
    prod_models = client.get_latest_versions(name=model_name, stages=["Production"])

    assert prod_models, f"No model version found in 'Production' for model '{model_name}'"

    latest_model = prod_models[0]
    model_uri = f"models:/{model_name}/{latest_model.version}"

    print(f"Latest Production Model URI: {model_uri}")
    print(f"Version: {latest_model.version}, Run ID: {latest_model.run_id}")

    # Load the model
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return loaded_model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_uri}: {e}")


# FastAPI app
app = FastAPI()

@app.on_event("startup")
def startup():
    global model
    print("*** startup event triggerred ***")
    model = load_production_model()

class CarFeatures(BaseModel):
    odometer: float
    fitnessAge: float
    featureCount: int
    make: str
    model: str
    variant: str
    year: int
    transmissionType: str
    bodyType: str
    fuelType: str
    ownership: str
    color: str

@app.get("/")
def home():
    return {"message": "Used Car Price Predictor API - Prashant Jain"}

@app.post("/predict")
def predict(data: CarFeatures):
    df = pd.DataFrame([data.dict()])
    try:
        pred = model.predict(df)
        return {"predicted_price": round(float(pred[0]), 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)