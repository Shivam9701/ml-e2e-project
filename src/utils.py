import os
import sys
import numpy as np
import dill
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging


def save_object(obj: object, file_path: str):
    """
    Save the object to the specified file path.
    Args:
        obj (object): The object to be saved.
        file_path (str): The file path to save the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)

        logging.info(f"Object saved to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Evaluate the models using the training and testing data.
    Args:
        models (dict): The models to be evaluated.
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training target.
        X_test (np.ndarray): The testing data.
        y_test (np.ndarray): The testing target.
    Returns:
        dict: The model report containing the scores of the models.
        Where the key is the model name and the value is the score.
    """

    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_pred)

            model_report[model_name] = test_score

    except Exception as e:
        raise CustomException(e, sys)
    return model_report
