import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame):
        try:
            logging.info("Loading model")
            model_path = "artifacts/model.pkl"
            model = load_object(model_path)
            logging.info("Model loaded successfully")

            logging.info("Loading the preprocessor")
            preprocessor_path = "artifacts/preprocessor.pkl"
            preprocessor = load_object(preprocessor_path)
            logging.info("Preprocessor loaded successfully")

            data_scaled = preprocessor.transform(data)
            prediction = model.predict(data_scaled)
            logging.info(f"Prediction: {prediction}")

            return prediction

        except Exception as e:
            raise CustomException(f"Error in prediction pipeline: {str(e)}", sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def to_dataframe(self):
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(
                f"Error in converting data to dataframe: {str(e)}", sys
            )
