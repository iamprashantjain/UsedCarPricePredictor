import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    

def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            logging.info(f"Object loaded successfully from: {file_path}")
            return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}")
        raise customexception(e, sys)


def register_and_promote_model(model_name: str, run_id: str, artifact_path: str, promote_to: str = "Staging"):
    """
    Utility to register a model and promote it to a given stage.

    Args:
        model_name (str): Name of the model in MLflow.
        run_id (str): Run ID from which the model artifact was generated.
        artifact_path (str): Path to the artifact relative to the run.
        promote_to (str): Stage to promote to ("Staging", "Production"). Default is "Staging".

    Raises:
        customexception: If any error occurs during the registration or promotion.
    """
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logging.info(f"Registering model from URI: {model_uri}")

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(f"Model registered with version {model_version.version}")

        # Transition stage
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=promote_to,
            archive_existing_versions=True
        )
        logging.info(f"Model version {model_version.version} promoted to '{promote_to}'.")

        # Optional: add tag for tracking
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="source",
            value="registered via utility"
        )

    except Exception as e:
        logging.error(f"Failed to register and promote model to {promote_to}.")
        raise customexception(e, sys)
