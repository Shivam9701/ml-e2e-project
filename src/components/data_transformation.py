import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Class responsible for data transformation.
    Args:
        config (DataTransformationConfig): The configuration object for data transformation.
    Methods:
        get_data_preprocessor(): Returns the preprocessor for data transformation.
    Raises:
        CustomException: If an error occurs during data transformation.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_preprocessor(self, numeric_features, categorical_features):
        """
        Returns the preprocessor for data transformation.
        Args:
            numeric_features (list): The list of numerical features.
            categorical_features (list): The list of categorical features
        Returns:
            preprocessor (ColumnTransformer): The preprocessor for data transformation.
        Raises:
            CustomException: If an error occurs during data transformation.
        """
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Numerical columns scaling pipeline created")
            logging.info("Categorical columns encoding pipeline created")

            preprocessor = ColumnTransformer(
                [
                    ("numeric_pipeline", numerical_pipeline, numeric_features),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_features,
                    ),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Initiates data transformation.
        Args:
            train_path (str): The path to the training dataset.
            test_path (str): The path to the testing dataset.
        Returns:
            train_arr (numpy.ndarray): The transformed training dataset.
            test_arr (numpy.ndarray): The transformed testing dataset.
            preprocessor_file_path (str): The path to the saved preprocessor.
        Raises:
            CustomException: If an error occurs during data transformation.
        """
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            # Select target column
            target_column_name = "math_score"

            # Drop target column from training and testing data
            input_train_data = train_data.drop(columns=[target_column_name])
            input_test_data = test_data.drop(columns=[target_column_name])

            # Select target column
            input_train_target = train_data[target_column_name]
            input_test_target = test_data[target_column_name]

            numeric_features = input_train_data.select_dtypes(
                include=[np.number]
            ).columns
            categorical_features = input_train_data.select_dtypes(
                include="object"
            ).columns

            preprocessor = self.get_data_preprocessor(
                numeric_features, categorical_features
            )

            logging.info(
                "Preprocessor created, fitting on training data and transforming testing data"
            )

            input_train_feature_arr = preprocessor.fit_transform(input_train_data)
            logging.info("Preprocessor fitted and applied on training data")

            input_test_feature_arr = preprocessor.transform(input_test_data)
            logging.info("Preprocessor transformed testing data")

            train_arr = np.c_[input_train_feature_arr, input_train_target]
            test_arr = np.c_[input_test_feature_arr, input_test_target]

            logging.info("Data transformation completed")
            logging.info("Saving preprocessor")

            save_object(preprocessor, self.config.preprocessor_file_path)

            return (train_arr, test_arr, self.config.preprocessor_file_path)

        except Exception as e:
            raise CustomException(e, sys)
