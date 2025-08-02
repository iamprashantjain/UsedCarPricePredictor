import sys
import os
import mlflow
import yaml
from src.logger.logging import logging
from src.exception.exception import customexception

class ModelPromoter:
    def __init__(self, params_path="params.yaml"):
        try:
            # Load parameters from the config file
            with open(params_path, "r") as f:
                self.params = yaml.safe_load(f)

            # Set model name and other required parameters
            self.model_name = self.params["model_registration"]["model_name"]
            self.repo_owner = self.params["mlflow"]["repo_owner"]
            self.repo_name = self.params["mlflow"]["repo_name"]
            

        except Exception as e:
            raise customexception(e, sys)

        # Authenticate to DagsHub using Personal Access Token (PAT)
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set MLflow tracking URI manually
        dagshub_url = "https://dagshub.com"
        mlflow.set_tracking_uri(f"{dagshub_url}/{self.repo_owner}/{self.repo_name}.mlflow")

        # MLflow Client for interaction with the model registry
        self.client = mlflow.MlflowClient()

    def get_latest_model_version(self, stage="Staging"):
        """
        Fetch the latest model version from a given stage (e.g., 'Staging').
        """
        versions = self.client.get_latest_versions(self.model_name, stages=[stage])
        return versions[0].version if versions else None

    def check_test_status(self):
        """
        Check the test result from the 'test_model_status.txt' file.
        """
        status_file_path = 'artifacts/test_model/test_model_status.txt'
        try:
            with open(status_file_path, 'r') as file:
                test_status = file.read().strip()
                if test_status == "Test PASSED":
                    logging.info("Test passed. Proceeding with model promotion.")
                    return True
                else:
                    logging.error("Test failed. Model promotion halted.")
                    return False
        except FileNotFoundError:
            logging.error(f"Test status file not found: {status_file_path}")
            raise Exception("Test status file missing. Cannot promote the model.")

    def promote_model(self):
        """
        Promote the latest model from 'Staging' to 'Production' in MLflow model registry.
        Archives any current 'Production' models before promotion.
        """
        # First, check if the tests passed before promoting
        if not self.check_test_status():
            logging.error("Tests did not pass. Skipping model promotion.")
            return

        # Proceed with model promotion if tests passed
        staging_version = self.get_latest_model_version(stage="Staging")
        if not staging_version:
            raise ValueError(f"No model version found in 'Staging' stage for {self.model_name}")

        # Archive current Production models (if any)
        prod_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
        if prod_versions:
            for version in prod_versions:
                logging.info(f"Archiving current Production version {version.version}")
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=version.version,
                    stage="Archived"
                )

        # Promote the Staging model to Production
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=staging_version,
            stage="Production"
        )
        logging.info(f"Model version {staging_version} promoted to Production successfully.")

if __name__ == "__main__":
    promoter = ModelPromoter()
    promoter.promote_model()
