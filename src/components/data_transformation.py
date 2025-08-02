import os
import sys
import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    transformed_train_path: str = os.path.join("artifacts", "data_transformation", "train_transformed.csv")
    transformed_test_path: str = os.path.join("artifacts", "data_transformation", "test_transformed.csv")
    preprocessor_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)

        self.numerical_cols = self.params["data_ingestion"]["numerical_cols"]
        self.categorical_cols = self.params["data_ingestion"]["categorical_cols"]
        self.target_column = self.params["base"]["target_col"]
        self.test_size = self.params["data_ingestion"]["test_size"]
        self.random_state = self.params["data_ingestion"]["random_state"]

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, self.numerical_cols),
                ("cat", cat_pipeline, self.categorical_cols)
            ])

            logging.info("Preprocessor pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise customexception(e, sys)

    def initiate_data_transformation(self):
        try:
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)

            # Read train and test CSVs
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            logging.info(f"Training Columns: {train_df.columns.tolist()}")
            logging.info(f"Target Column: {self.target_column}")

            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]

            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Check for columns mismatch
            if X_train_transformed.shape[1] != X_test_transformed.shape[1]:
                raise ValueError(f"Feature mismatch: Train shape: {X_train_transformed.shape[1]}, Test shape: {X_test_transformed.shape[1]}")

            # Save transformed arrays as CSVs
            pd.DataFrame(X_train_transformed).to_csv(self.config.transformed_train_path, index=False)
            pd.DataFrame(X_test_transformed).to_csv(self.config.transformed_test_path, index=False)

            # Save preprocessor
            save_object(file_path=self.config.preprocessor_path, obj=preprocessor)
            logging.info(f"Preprocessor saved at: {self.config.preprocessor_path}")

            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            logging.error("Error in initiate_data_transformation")
            raise customexception(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation()
