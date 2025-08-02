# import os
# import sys
# import yaml
# import pandas as pd
# import numpy as np
# import mlflow
# import mlflow.sklearn
# from dataclasses import dataclass
# from xgboost import XGBRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import r2_score, mean_squared_error

# from src.logger.logging import logging
# from src.exception.exception import customexception
# from src.utils.utils import save_object, load_object


# @dataclass
# class ModelTrainerConfig:
#     train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
#     test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
#     preprocessor_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
#     model_path: str = os.path.join("artifacts", "model_trainer", "model.pkl")


# class ModelTrainer:
#     def __init__(self):
#         self.config = ModelTrainerConfig()

#         with open("params.yaml", "r") as f:
#             self.params = yaml.safe_load(f)

#         self.target_column = self.params["base"]["target_col"]
#         self.model_params = self.params["model_trainer"]  # Load all model params directly
#         self.repo_owner = self.params["mlflow"]["repo_owner"]
#         self.repo_name = self.params["mlflow"]["repo_name"]

#         # Set up DagsHub credentials for MLflow tracking
#         dagshub_token = os.getenv("DAGSHUB_PAT")
#         if not dagshub_token:
#             raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

#         os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#         # MLflow tracking URI for DagsHub
#         dagshub_url = "https://dagshub.com"
#         mlflow.set_tracking_uri(f'{dagshub_url}/{self.repo_owner}/{self.repo_name}.mlflow')    

#     def initiate_model_training(self):
#         try:
#             logging.info("Model training started using preprocessor pipeline.")

#             # Load raw training and test data
#             train_df = pd.read_csv(self.config.train_data_path)
#             test_df = pd.read_csv(self.config.test_data_path)

#             X_train = train_df.drop(columns=[self.target_column])
#             y_train = train_df[self.target_column]

#             X_test = test_df.drop(columns=[self.target_column])
#             y_test = test_df[self.target_column]

#             # Load preprocessor object
#             preprocessor = load_object(self.config.preprocessor_path)
#             logging.info("Preprocessor loaded successfully.")

#             # Create full pipeline with preprocessor + model
#             model = XGBRegressor(**self.model_params)
#             pipeline = Pipeline(steps=[
#                 ("preprocessor", preprocessor),
#                 ("regressor", model)
#             ])

#             # Start an MLflow run and log to DagsHub
#             with mlflow.start_run() as run:
#                 # Log model parameters
#                 for param, value in self.model_params.items():
#                     mlflow.log_param(param, value)

#                 # Train full pipeline on raw training data
#                 pipeline.fit(X_train, y_train)
#                 logging.info("Model pipeline trained successfully.")

#                 # Predict on test data
#                 y_pred = pipeline.predict(X_test)

#                 # Evaluate performance
#                 r2 = r2_score(y_test, y_pred)
#                 rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#                 logging.info(f"Current R2 Score: {r2:.4f}, RMSE: {rmse:.2f}")

#                 # Log metrics
#                 mlflow.log_metric("r2_score", r2)
#                 mlflow.log_metric("rmse", rmse)

#                 # Log the model to MLflow hosted on DagsHub
#                 mlflow.sklearn.log_model(pipeline, "model")

#                 # Optionally, save the model locally
#                 os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
#                 save_object(file_path=self.config.model_path, obj=pipeline)
#                 logging.info(f"Model pipeline saved at: {self.config.model_path}")

#                 return pipeline, r2, rmse

#         except Exception as e:
#             logging.error("Error occurred during model training.")
#             raise customexception(e, sys)


# if __name__ == "__main__":
#     obj = ModelTrainer()
#     obj.initiate_model_training()




import os
import sys
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object, load_object


@dataclass
class ModelTrainerConfig:
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    preprocessor_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
    model_path: str = os.path.join("artifacts", "model_trainer", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)

        self.target_column = self.params["base"]["target_col"]
        self.model_params = self.params["model_trainer"]
        self.repo_owner = self.params["mlflow"]["repo_owner"]
        self.repo_name = self.params["mlflow"]["repo_name"]
        self.model_name = self.params["model_registration"]["model_name"]

        # Set up DagsHub token for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        mlflow.set_tracking_uri(f'{dagshub_url}/{self.repo_owner}/{self.repo_name}.mlflow')

    def initiate_model_training(self):
        try:
            logging.info("Model training started.")

            # Load training and test data
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            # Load the preprocessor
            preprocessor = load_object(self.config.preprocessor_path)
            logging.info("Preprocessor loaded.")

            # Create pipeline
            model = XGBRegressor(**self.model_params)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])

            # Start MLflow run
            with mlflow.start_run() as run:
                # Log parameters
                for param, value in self.model_params.items():
                    mlflow.log_param(param, value)

                # Train model
                pipeline.fit(X_train, y_train)
                logging.info("Pipeline training complete.")

                # Predictions & Metrics
                y_pred = pipeline.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)

                logging.info(f"R2: {r2:.4f}, RMSE: {rmse:.2f}")

                # Register model (to 'None' stage by default)
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name=self.model_name
                )

                logging.info(f"Model registered under name: {self.model_name} (stage: None)")

                # Save locally
                os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
                save_object(file_path=self.config.model_path, obj=pipeline)
                logging.info(f"Model saved to: {self.config.model_path}")

                return pipeline, r2, rmse

        except Exception as e:
            logging.error("Exception during model training.")
            raise customexception(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()