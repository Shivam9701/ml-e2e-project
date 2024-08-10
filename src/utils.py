import os
import sys
import numpy as np
import pandas as pd
import dill


from src.exception import CustomException
from src.logger import logging


def save_object(obj, file_path):
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
