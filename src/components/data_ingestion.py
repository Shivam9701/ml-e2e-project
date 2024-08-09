import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data", "raw.csv")
    training_data_path: str = os.path.join("artifacts", "data", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data", "test.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")
            self.load_data()
            self.split_data()
            self.save_data()
            logging.info("Data ingestion completed")
            return (self.config.training_data_path, self.config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        try:
            logging.info("Loading data")
            self.df = pd.read_csv("notebooks\data\stud.csv")
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self):
        try:
            logging.info("Splitting data")
            self.train_set, self.test_set = train_test_split(
                self.df, test_size=0.2, random_state=42
            )
            logging.info("Train test split completed")
        except Exception as e:
            raise CustomException(e, sys)

    def save_data(self):
        try:
            logging.info("Saving data")
            os.makedirs(os.path.dirname(self.config.training_data_path), exist_ok=True)
            self.df.to_csv(self.config.raw_data_path, index=False)
            self.train_set.to_csv(self.config.training_data_path, index=False)
            self.test_set.to_csv(self.config.test_data_path, index=False)
            logging.info("Data saved")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    data_ingestion.initiate_data_ingestion()
