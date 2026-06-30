import os
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logger
from src.exception import CustomException


class DataTransformation:

    def __init__(self):
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def initiate_data_transformation(self, train_path, test_path):

        try:

            logger.info("Starting Data Transformation")

            # Read Train and Test Dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "price"

            # Split Features and Target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Identify Numerical Columns
            numerical_columns = X_train.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            # Identify Categorical Columns
            categorical_columns = X_train.select_dtypes(
                include=["object"]
            ).columns.tolist()

            logger.info(f"Numerical Columns: {numerical_columns}")
            logger.info(f"Categorical Columns: {categorical_columns}")

            # Numerical Pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            # Combine Both Pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            logger.info("Applying Data Transformation")

            # Transform Training Data
            X_train = preprocessor.fit_transform(X_train)

            # Transform Testing Data
            X_test = preprocessor.transform(X_test)

            # Combine Features and Target
            train_array = np.c_[X_train, np.array(y_train)]
            test_array = np.c_[X_test, np.array(y_test)]

            # Create Artifacts Folder
            os.makedirs("artifacts", exist_ok=True)

            # Save Preprocessor
            with open(self.preprocessor_path, "wb") as file:
                pickle.dump(preprocessor, file)

            logger.info("Preprocessor Saved Successfully")
            logger.info("Data Transformation Completed Successfully")

            return (
                train_array,
                test_array,
                self.preprocessor_path
            )

        except Exception as e:

            logger.error("Error During Data Transformation")

            raise CustomException(e, sys)