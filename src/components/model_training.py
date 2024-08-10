import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainingConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.models = {
            "LinearRegression": LinearRegression(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "XGBRegressor": XGBRegressor(),
        }

    def initiate_model_trainer(self, train_arr, test_arr):
        logging.info("Initiating Model Training")
        try:
            X_train, y_train = (train_arr[:, :-1], train_arr[:, -1])
            X_test, y_test = (test_arr[:, :-1], test_arr[:, -1])

            model_report: dict = evaluate_models(
                models=self.models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            report_sorted = dict(
                sorted(model_report.items(), key=lambda item: item[1], reverse=True)
            )

            best_model_name, best_model_score = list(report_sorted.items())[0]

            best_model = self.models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model has a score greater than 0.6", sys)

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Model Score: {best_model_score}")

            logging.info("Saving the best model")
            save_object(best_model, self.config.model_file_path)

            logging.info("Model Training Completed")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
