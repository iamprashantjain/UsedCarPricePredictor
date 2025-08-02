import yaml
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger.logging import logging
from src.exception.exception import customexception


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")


class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ingestion_config = DataIngestionConfig()
        
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)
        
        self.test_size = self.params["data_ingestion"]["test_size"]
        self.random_state = self.params["data_ingestion"]["random_state"]
        self.numerical_cols = self.params["data_ingestion"]["numerical_cols"]
        self.categorical_cols = self.params["data_ingestion"]["categorical_cols"]

    def initial_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("initial_data_cleaning started")
        df.drop_duplicates(inplace=True)
        df = df[self.numerical_cols + self.categorical_cols + [self.params["base"]["target_col"]]]  # <-- Add target column
        return df


    def initiate_data_ingestion(self):
        logging.info("initiate_data_ingestion started")
        try:
            data = pd.read_excel(self.file_path)
            logging.info(f"Data fetched successfully from {self.file_path}")

            data = self.initial_data_cleaning(data)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed and train-test split saved.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    obj = DataIngestion(file_path='experiment/cars24_v3.xlsx')
    obj.initiate_data_ingestion()