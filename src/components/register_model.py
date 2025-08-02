# import os
# import sys
# import json
# import yaml
# import mlflow
# from mlflow.tracking import MlflowClient
# from dataclasses import dataclass

# from src.logger.logging import logging
# from src.exception.exception import customexception


# @dataclass
# class ModelRegistrationConfig:
#     params_path: str = "params.yaml"
#     model_info_path: str = os.path.join("artifacts", "model_evaluation", "evaluation_info.json")
#     model_name: str = None


# class ModelRegistration:
#     def __init__(self):
#         try:
#             self.config = ModelRegistrationConfig()

#             # Load model_name from params.yaml
#             with open(self.config.params_path, 'r') as f:
#                 params = yaml.safe_load(f)
            
#             self.config.model_name = params["model_registration"]["model_name"]
#             self.config.repo_owner = params["mlflow"]["repo_owner"]
#             self.config.repo_name = params["mlflow"]["repo_name"]
            
#             # Set up DagsHub credentials for MLflow tracking
#             dagshub_token = os.getenv("DAGSHUB_PAT")
#             if not dagshub_token:
#                 raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

#             os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#             os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#             # MLflow tracking URI for DagsHub
#             dagshub_url = "https://dagshub.com"
#             mlflow.set_tracking_uri(f'{dagshub_url}/{self.config.repo_owner}/{self.config.repo_name}.mlflow')

#         except Exception as e:
#             logging.error("Failed to load or parse model_name from params.yaml")
#             raise customexception(e, sys)

#     def _load_model_info(self):
#         try:
#             with open(self.config.model_info_path, 'r') as f:
#                 model_info = json.load(f)
#             logging.info(f"Loaded model info from {self.config.model_info_path}")
#             return model_info
#         except Exception as e:
#             logging.error("Failed to load model info JSON.")
#             raise customexception(e, sys)

#     def register_model(self):
#         try:
#             model_info = self._load_model_info()

#             # Check the 'register' flag before proceeding
#             if not model_info.get("register", False):
#                 logging.info("Model registration flag is set to False. Skipping registration.")
#                 return

#             model_uri = f"runs:/{model_info['run_id']}/{model_info['artifact_path']}"

#             # Register the model with MLflow
#             model_version = mlflow.register_model(model_uri, self.config.model_name)
#             logging.info(f"Model registered with version {model_version.version}")

#             # Transition the model version to 'Staging'
#             client = MlflowClient()

#             model = client.get_registered_model(self.config.model_name)
#             model_version = client.get_model_version(self.config.model_name, model_version.version)

#             if model_version.current_stage == "None":
#                 logging.info(f"Transitioning model version {model_version.version} to 'Staging'...")
#                 client.transition_model_version_stage(
#                     name=self.config.model_name,
#                     version=model_version.version,
#                     stage="Staging"
#                 )
#                 logging.info(f"Model version {model_version.version} successfully moved to 'Staging'.")
#             else:
#                 logging.info(f"Model version {model_version.version} is already in stage {model_version.current_stage}.")

#         except Exception as e:
#             logging.error("Model registration failed.")
#             raise customexception(e, sys)



# if __name__ == "__main__":
#     registration = ModelRegistration()
#     registration.register_model()



import os
import sys
import json
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import dataclass

from src.logger.logging import logging
from src.exception.exception import customexception


@dataclass
class ModelRegistrationConfig:
    params_path: str = "params.yaml"
    model_info_path: str = os.path.join("artifacts", "model_evaluation", "evaluation_info.json")
    model_name: str = None


class ModelRegistration:
    def __init__(self):
        try:
            self.config = ModelRegistrationConfig()

            # Load params.yaml
            with open(self.config.params_path, 'r') as f:
                params = yaml.safe_load(f)
            
            self.config.model_name = params["model_registration"]["model_name"]
            self.config.repo_owner = params["mlflow"]["repo_owner"]
            self.config.repo_name = params["mlflow"]["repo_name"]
            
            # DAGsHub credentials
            dagshub_token = os.getenv("DAGSHUB_PAT")
            if not dagshub_token:
                raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            mlflow.set_tracking_uri(f"https://dagshub.com/{self.config.repo_owner}/{self.config.repo_name}.mlflow")

        except Exception as e:
            logging.error("Failed to initialize ModelRegistration.")
            raise customexception(e, sys)

    def _load_model_info(self):
        try:
            with open(self.config.model_info_path, 'r') as f:
                model_info = json.load(f)
            logging.info(f"Loaded model info from {self.config.model_info_path}")
            return model_info
        except Exception as e:
            logging.error("Failed to load model info JSON.")
            raise customexception(e, sys)

    def register_model(self):
        try:
            model_info = self._load_model_info()

            if not model_info.get("register", False):
                logging.info("Model registration skipped as flag is set to False.")
                return

            model_uri = f"runs:/{model_info['run_id']}/{model_info['artifact_path']}"
            logging.info(f"Registering model from URI: {model_uri}")

            # Register model to None
            model_version = mlflow.register_model(model_uri, self.config.model_name)
            logging.info(f"Model registered to None with version {model_version.version}")

            # Transition to Staging
            client = MlflowClient()
            client.transition_model_version_stage(
                name=self.config.model_name,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=True  # This line archives older staging versions
            )
            logging.info(f"Model version {model_version.version} moved to 'Staging'.")

            # Optionally: Add a tag or description
            client.set_model_version_tag(
                name=self.config.model_name,
                version=model_version.version,
                key="source",
                value="automated registration pipeline"
            )

        except Exception as e:
            logging.error("Model registration failed.")
            raise customexception(e, sys)


if __name__ == "__main__":
    registration = ModelRegistration()
    registration.register_model()
