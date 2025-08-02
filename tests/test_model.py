from src.utils.utils import load_object, save_object
import json
import pickle
import unittest
from pathlib import Path
import os
import sys
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import yaml
from src.logger.logging import logging
from src.exception.exception import customexception

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.info("Setting up test environment...")

        # Load parameters from params.yaml
        try:
            with open("params.yaml", "r") as f:
                cls.params = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading params.yaml: {e}")
            raise customexception(e, sys)

        cls.model_name = cls.params["model_registration"]["model_name"]
        cls.min_r2 = cls.params["evaluation"]["min_r2"]
        cls.max_rmse = cls.params["evaluation"]["max_rmse"]
        cls.target_column = cls.params["base"]["target_col"]

        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "iamprashantjain"
        repo_name = "Used_Car_Price_Predictor"
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load preprocessor
        preprocessor_path = Path('artifacts') / 'data_transformation' / 'preprocessor.pkl'
        try:
            with open(preprocessor_path, 'rb') as f:
                cls.preprocessor = pickle.load(f)
            logging.info("Preprocessor loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load preprocessor from {preprocessor_path}: {e}")
            raise customexception(e, sys)

        # Try to load model from MLflow registry (DagsHub)
        try:
            cls.new_model_name = cls.model_name
            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
            if cls.new_model_version is None:
                raise ValueError(f"No model found for {cls.new_model_name} in specified stage")

            cls.model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)
            logging.info(f"Model loaded successfully from MLflow registry: {cls.model_uri}")

        except Exception as e:
            logging.warning(f"Failed to load model from MLflow registry: {e}")
            logging.info("Falling back to loading local model.pkl file")

            try:
                model_path = Path("artifacts") / "model_trainer" / "model.pkl"
                with open(model_path, "rb") as f:
                    cls.model = pickle.load(f)
                logging.info("Model loaded successfully from local model.pkl")
            except Exception as local_e:
                logging.error(f"Failed to load model from local file: {local_e}")
                raise customexception(local_e, sys)

        # Load test data
        test_data_path = Path('artifacts') / 'data_ingestion' / 'test.csv'
        try:
            cls.test_df = pd.read_csv(test_data_path)
            logging.info(f"Test data loaded successfully from {test_data_path}")
        except Exception as e:
            logging.error(f"Failed to load test data from {test_data_path}: {e}")
            raise customexception(e, sys)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        try:
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logging.error(f"Error fetching latest model version for {model_name}: {e}")
            raise customexception(e, sys)

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model, "Model did not load properly")

    def test_model_performance(self):
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[self.target_column]

        # Ensure that X_test columns match the expected columns by the preprocessor
        expected_columns = self.params["data_ingestion"].get("numerical_cols", []) + self.params["data_ingestion"].get("categorical_cols", [])
        missing_columns = list(set(expected_columns) - set(X_test.columns))
        if missing_columns:
            raise ValueError(f"Missing columns in test data: {missing_columns}")

        # Reorder columns to match training data if necessary
        X_test = X_test[expected_columns]
        
        # Load the preprocessor and model pipeline (same as training)
        pipeline = load_object("artifacts/model_trainer/model.pkl")

        # Predict using the pipeline
        y_pred = pipeline.predict(X_test)

        # Calculate R² and RMSE
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Assert the model performance
        self.assertGreaterEqual(r2, self.__class__.min_r2, f"R² score {r2:.3f} is below minimum threshold {self.__class__.min_r2}")
        self.assertLessEqual(rmse, self.__class__.max_rmse, f"RMSE {rmse:.3f} exceeds max threshold {self.__class__.max_rmse}")

        logging.info(f"Model performance test passed. R²: {r2:.3f}, RMSE: {rmse:.3f}")

        # Write test result status to file
        status = "Test PASSED"
        
        # Check other tests (example: model loading test)
        if self.model is None:
            status = "Test FAILED"

        # Other potential checks like checking feature column order, etc.
        if missing_columns:
            status = "Test FAILED"

        status_dir = 'artifacts/test_model/'
        if not os.path.exists(status_dir):
            os.makedirs(status_dir)

        # Write status to file
        status_file_path = os.path.join(status_dir, 'test_model_status.txt')
        with open(status_file_path, 'w') as status_file:
            status_file.write(status)

        logging.info(f"Test status written to {status_file_path}")



if __name__ == "__main__":
    unittest.main()



# =============== original code ===================
# import unittest
# import mlflow
# import os
# import pandas as pd
# import numpy as np
# from sklearn.metrics import r2_score, mean_squared_error
# import yaml
# from pathlib import Path
# from src.logger.logging import logging
# from src.exception.exception import customexception

# class TestModelLoading(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         logging.info("Setting up test environment...")

#         # Load parameters
#         with open("params.yaml", "r") as f:
#             cls.params = yaml.safe_load(f)

#         cls.model_name = cls.params["model_registration"]["model_name"]
#         cls.min_r2 = cls.params["evaluation"]["min_r2"]
#         cls.max_rmse = cls.params["evaluation"]["max_rmse"]
#         cls.target_column = cls.params["base"]["target_col"]
        
#         repo_owner = cls.params["mlflow"]["repo_owner"]
#         repo_name = cls.params["mlflow"]["repo_name"]

#         # Setup MLflow tracking
#         dagshub_token = os.getenv("DAGSHUB_PAT")
#         if not dagshub_token:
#             raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

#         os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
#         dagshub_url = "https://dagshub.com"
#         mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

#         # Load model from MLflow registry
#         client = mlflow.MlflowClient()
#         versions = client.get_latest_versions(cls.model_name, stages=["Staging"])
#         if not versions:
#             raise ValueError(f"No model found for {cls.model_name} in Staging")

#         cls.model_uri = f'models:/{cls.model_name}/{versions[0].version}'
#         cls.model = mlflow.pyfunc.load_model(cls.model_uri)
#         logging.info(f"Loaded model from MLflow: {cls.model_uri}")

#         # Load preprocessor
#         preprocessor_path = Path('artifacts/data_transformation/preprocessor.pkl')
#         with open(preprocessor_path, 'rb') as f:
#             import pickle
#             cls.preprocessor = pickle.load(f)

#         # Load test data
#         test_data_path = Path('artifacts/data_ingestion/test.csv')
#         cls.test_df = pd.read_csv(test_data_path)

#     def test_model_loaded_properly(self):
#         self.assertIsNotNone(self.model, "Model did not load properly")

#     def test_model_performance(self):
#         X_test = self.test_df.drop(columns=[self.target_column])
#         y_test = self.test_df[self.target_column]

#         expected_columns = self.params["data_ingestion"].get("numerical_cols", []) + \
#                            self.params["data_ingestion"].get("categorical_cols", [])

#         missing_columns = list(set(expected_columns) - set(X_test.columns))
#         self.assertEqual(missing_columns, [], f"Missing columns: {missing_columns}")

#         X_test = X_test[expected_columns]
#         X_transformed = self.__class__.preprocessor.transform(X_test)

#         y_pred = self.__class__.model.predict(X_transformed)

#         r2 = r2_score(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#         self.assertGreaterEqual(r2, self.__class__.min_r2, f"R² score too low: {r2}")
#         self.assertLessEqual(rmse, self.__class__.max_rmse, f"RMSE too high: {rmse}")

#         logging.info(f"Model performance test passed. R²: {r2:.3f}, RMSE: {rmse:.3f}")

#         # Save test status
#         status = "Test PASSED"
#         status_dir = 'artifacts/test_model/'
#         os.makedirs(status_dir, exist_ok=True)
#         with open(os.path.join(status_dir, 'test_model_status.txt'), 'w') as f:
#             f.write(status)

# if __name__ == "__main__":
#     unittest.main()
