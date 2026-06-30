import os
import sys
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.logger import logger
from src.exception import CustomException


class ModelTrainer:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logger.info("Starting Model Training")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Linear Regression": LinearRegression(),

                "Decision Tree": DecisionTreeRegressor(random_state=42),

                "Random Forest": RandomForestRegressor(random_state=42)

            }

            best_score = -1
            best_model = None
            best_model_name = None

            for model_name, model in models.items():

                model.fit(X_train, y_train)

                prediction = model.predict(X_test)

                score = r2_score(y_test, prediction)

                logger.info(f"{model_name} R2 Score : {score}")

                if score > best_score:

                    best_score = score

                    best_model = model

                    best_model_name = model_name

            os.makedirs("artifacts", exist_ok=True)

            with open(self.model_path, "wb") as file:

                pickle.dump(best_model, file)

            logger.info(f"Best Model : {best_model_name}")

            logger.info(f"Best Score : {best_score}")

            return best_model_name, best_score

        except Exception as e:

            logger.error("Error During Model Training")

            raise CustomException(e, sys)