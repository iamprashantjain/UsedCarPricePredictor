# import os
# import sys
# import json
# import yaml
# import mlflow
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from sklearn.metrics import r2_score, mean_squared_error

# from src.logger.logging import logging
# from src.exception.exception import customexception
# from src.utils.utils import load_object, save_object
# import yaml

# with open("params.yaml", "r") as f:
#     params = yaml.safe_load(f)

# repo_owner = params["mlflow"]["repo_owner"]
# repo_name = params["mlflow"]["repo_name"]

# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# @dataclass
# class ModelEvaluationConfig:
#     test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
#     model_path: str = os.path.join("artifacts", "model_trainer", "model.pkl")
#     metrics_path: str = os.path.join("artifacts", "model_evaluation", "metrics.yaml")
#     evaluation_info_path: str = os.path.join("artifacts", "model_evaluation", "evaluation_info.json")
#     best_model_path: str = os.path.join("artifacts", "model_evaluation", "best_model.pkl")
#     experiment_name: str = "Used_Car_Price_Predictor_Experiment"
#     run_name: str = "Model_Evaluation_Run"


# class ModelEvaluation:
#     def __init__(self):
#         self.config = ModelEvaluationConfig()
#         try:
#             with open("params.yaml", "r") as f:
#                 self.params = yaml.safe_load(f)
#             self.target_column = self.params["base"]["target_col"]
#         except Exception as e:
#             logging.error("Failed to load params.yaml")
#             raise customexception(e, sys)

#     def load_test_data(self):
#         try:
#             df = pd.read_csv(self.config.test_data_path)
#             X_test = df.drop(columns=[self.target_column])
#             y_test = df[self.target_column]
#             logging.info("Test data loaded successfully.")
#             return X_test, y_test
#         except Exception as e:
#             logging.error("Failed to load test data")
#             raise customexception(e, sys)

#     def evaluate_model(self, pipeline, X_test, y_test):
#         try:
#             # Predict using the model in the pipeline (this includes preprocessing and prediction)
#             y_pred = pipeline.predict(X_test)
            
#             # Evaluate performance
#             r2 = r2_score(y_test, y_pred)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             logging.info(f"Evaluation complete. R2: {r2:.4f}, RMSE: {rmse:.2f}")
#             return r2, rmse
#         except Exception as e:
#             logging.error("Error during model evaluation")
#             raise customexception(e, sys)
    
    
#     def save_metrics(self, r2, rmse):
#         try:
#             os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
#             metrics = {"r2_score": float(r2), "rmse": float(rmse)}
#             with open(self.config.metrics_path, "w") as f:
#                 yaml.dump(metrics, f)
#             logging.info(f"Metrics saved at {self.config.metrics_path}")
#         except Exception as e:
#             logging.error("Failed to save evaluation metrics")
#             raise customexception(e, sys)

#     def save_best_model(self, model, r2, best_score):
#         try:
#             # First model scenario or better score
#             if best_score is None or r2 > best_score:
#                 logging.info(f"Saving model with R2: {r2}")
#                 os.makedirs(os.path.dirname(self.config.best_model_path), exist_ok=True)
#                 save_object(self.config.best_model_path, model)

#                 # Log the artifact path where the model is stored
#                 artifact_path = self.config.best_model_path  # Save this path for registration later
#                 logging.info(f"Model artifact saved at {artifact_path}")
#             else:
#                 logging.info(f"Model with R2: {r2} is not better than the previous best score: {best_score}. Not saving.")
#                 artifact_path = None  # If the model is not better, set artifact path to None
#         except Exception as e:
#             logging.error("Failed to save the best model")
#             raise customexception(e, sys)
#         return artifact_path

#     def save_evaluation_info(self, run_id, register_flag, artifact_path):
#         try:
#             os.makedirs(os.path.dirname(self.config.evaluation_info_path), exist_ok=True)
#             evaluation_info = {
#                 "run_id": run_id,
#                 "metrics_path": self.config.metrics_path,
#                 "register": bool(register_flag),
#                 "artifact_path": artifact_path  # Include the artifact path
#             }
#             with open(self.config.evaluation_info_path, "w") as f:
#                 json.dump(evaluation_info, f, indent=4)
#             logging.info(f"Evaluation info saved at {self.config.evaluation_info_path}")
#         except Exception as e:
#             logging.error("Failed to save evaluation info")
#             raise customexception(e, sys)

#     def get_previous_best_score(self):
#         try:
#             if not os.path.exists(self.config.metrics_path):
#                 logging.info("No previous metrics found. Assuming this is the first model.")
#                 return None

#             with open(self.config.metrics_path, "r") as f:
#                 metrics = yaml.safe_load(f)
#             return metrics.get("r2_score")
#         except Exception as e:
#             logging.error("Failed to load previous metrics for comparison")
#             raise customexception(e, sys)

#     def initiate_model_evaluation(self):
#         try:
#             logging.info("Model evaluation process started...")

#             # Load the test data
#             X_test, y_test = self.load_test_data()
#             pipeline = load_object(self.config.model_path)  # Load the entire pipeline
            
#             logging.info("Pipeline loaded successfully.")

#             # Set MLFlow experiment
#             mlflow.set_experiment(self.config.experiment_name)

#             with mlflow.start_run(run_name=self.config.run_name) as run:
#                 # Evaluate the model using the pipeline
#                 r2, rmse = self.evaluate_model(pipeline, X_test, y_test)

#                 # Log metrics to MLFlow
#                 mlflow.log_metric("r2_score", r2)
#                 mlflow.log_metric("rmse", rmse)               
                
#                 #log model
#                 mlflow.sklearn.log_model(pipeline, "model")

#                 # Get the previous best score from the metrics.yaml
#                 best_score = self.get_previous_best_score()

#                 # Check if this is the first model or if the new model is better
#                 if best_score is None:
#                     logging.info("First model, automatically eligible for registration.")
#                     register_flag = True  # Register the first model
#                 else:
#                     logging.info(f"Previous best score: {best_score}")
#                     register_flag = r2 > best_score
#                     if register_flag:
#                         logging.info(f"New model is better (R2: {r2} > {best_score}), will register.")
#                     else:
#                         logging.info(f"New model is worse (R2: {r2} <= {best_score}), will not register.")

#                 # Save metrics and model (if necessary)
#                 self.save_metrics(r2, rmse)
#                 artifact_path = self.save_best_model(pipeline, r2, best_score)

#                 # Save evaluation info
#                 self.save_evaluation_info(run.info.run_id, register_flag, artifact_path)

#             logging.info("Model evaluation completed successfully.")
#             return r2, rmse, register_flag

#         except Exception as e:
#             logging.error("Error in model evaluation pipeline")
#             raise customexception(e, sys)


# if __name__ == "__main__":
#     evaluator = ModelEvaluation()
#     evaluator.initiate_model_evaluation()



# =============================


import os
import sys
import json
import yaml
import mlflow
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error

from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import load_object, save_object


# Load config from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

repo_owner = params["mlflow"]["repo_owner"]
repo_name = params["mlflow"]["repo_name"]

# Set up MLflow to use DagsHub
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")


@dataclass
class ModelEvaluationConfig:
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    model_path: str = os.path.join("artifacts", "model_trainer", "model.pkl")
    metrics_path: str = os.path.join("artifacts", "model_evaluation", "metrics.yaml")
    evaluation_info_path: str = os.path.join("artifacts", "model_evaluation", "evaluation_info.json")
    best_model_local_path: str = os.path.join("artifacts", "model_evaluation", "best_model.pkl")
    experiment_name: str = "Used_Car_Price_Predictor_Experiment"
    run_name: str = "Model_Evaluation_Run"


class ModelEvaluation:
    def __init__(self):
        self.config = ModelEvaluationConfig()
        try:
            with open("params.yaml", "r") as f:
                self.params = yaml.safe_load(f)
            self.target_column = self.params["base"]["target_col"]
        except Exception as e:
            raise customexception(e, sys)

    def load_test_data(self):
        try:
            df = pd.read_csv(self.config.test_data_path)
            X_test = df.drop(columns=[self.target_column])
            y_test = df[self.target_column]
            return X_test, y_test
        except Exception as e:
            raise customexception(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return r2, rmse
        except Exception as e:
            raise customexception(e, sys)

    def save_metrics(self, r2, rmse):
        try:
            os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
            with open(self.config.metrics_path, "w") as f:
                yaml.dump({"r2_score": float(r2), "rmse": float(rmse)}, f)
        except Exception as e:
            raise customexception(e, sys)

    def save_best_model_locally(self, model, r2, best_score):
        try:
            if best_score is None or r2 > best_score:
                os.makedirs(os.path.dirname(self.config.best_model_local_path), exist_ok=True)
                save_object(self.config.best_model_local_path, model)
                return True
            return False
        except Exception as e:
            raise customexception(e, sys)

    def get_previous_best_score(self):
        try:
            if not os.path.exists(self.config.metrics_path):
                return None
            with open(self.config.metrics_path, "r") as f:
                return yaml.safe_load(f).get("r2_score")
        except Exception as e:
            raise customexception(e, sys)

    def save_evaluation_info(self, run_id, registered):
        try:
            os.makedirs(os.path.dirname(self.config.evaluation_info_path), exist_ok=True)
            with open(self.config.evaluation_info_path, "w") as f:
                json.dump({
                    "run_id": run_id,
                    "metrics_path": self.config.metrics_path,
                    "model_registered": registered
                }, f, indent=4)
        except Exception as e:
            raise customexception(e, sys)

    def initiate_model_evaluation(self):
        try:
            logging.info("Starting model evaluation...")
            X_test, y_test = self.load_test_data()
            model = load_object(self.config.model_path)

            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run(run_name=self.config.run_name) as run:
                r2, rmse = self.evaluate_model(model, X_test, y_test)

                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)

                mlflow.sklearn.log_model(model, "model")

                best_score = self.get_previous_best_score()
                is_better = self.save_best_model_locally(model, r2, best_score)

                self.save_metrics(r2, rmse)

                registered = False
                if is_better:
                    logging.info("New model is better. Registering...")
                    mlflow.register_model(
                        model_uri=f"runs:/{run.info.run_id}/model",
                        name="UsedCarPriceModel"
                    )
                    registered = True
                else:
                    logging.info("New model is not better. Skipping registration.")

                self.save_evaluation_info(run.info.run_id, registered)

            logging.info("Model evaluation completed successfully.")
            return r2, rmse, registered

        except Exception as e:
            raise customexception(e, sys)


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    r2, rmse, registered = evaluator.initiate_model_evaluation()
    print(f"R2: {r2:.4f}, RMSE: {rmse:.2f}, Registered: {registered}")
