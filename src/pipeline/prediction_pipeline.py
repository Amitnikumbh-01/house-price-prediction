import pickle
import sys

from src.logger import logger
from src.exception import CustomException


class PredictionPipeline:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features):

        try:

            with open(self.model_path, "rb") as model_file:
                model = pickle.load(model_file)

            with open(self.preprocessor_path, "rb") as preprocessor_file:
                preprocessor = pickle.load(preprocessor_file)

            transformed_data = preprocessor.transform(features)

            prediction = model.predict(transformed_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)