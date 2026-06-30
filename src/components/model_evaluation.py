import sys
import pickle

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.logger import logger
from src.exception import CustomException


class ModelEvaluation:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def initiate_model_evaluation(
        self,
        test_array
    ):

        try:

            logger.info("Starting Model Evaluation")

            with open(self.model_path, "rb") as file:

                model = pickle.load(file)

            X_test = test_array[:, :-1]

            y_test = test_array[:, -1]

            prediction = model.predict(X_test)

            mae = mean_absolute_error(
                y_test,
                prediction
            )

            mse = mean_squared_error(
                y_test,
                prediction
            )

            rmse = mse ** 0.5

            r2 = r2_score(
                y_test,
                prediction
            )

            logger.info(f"MAE : {mae}")
            logger.info(f"MSE : {mse}")
            logger.info(f"RMSE : {rmse}")
            logger.info(f"R2 Score : {r2}")

            return {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2
            }

        except Exception as e:

            logger.error("Error During Model Evaluation")

            raise CustomException(e, sys)