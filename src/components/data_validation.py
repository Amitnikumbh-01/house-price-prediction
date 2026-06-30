import sys

import pandas as pd

from src.logger import logger
from src.exception import CustomException


class DataValidation:

    def __init__(self):
        pass

    def validate_dataset(self, file_path):

        try:

            logger.info("Starting Data Validation")

            df = pd.read_csv(file_path)

            if df.empty:
                raise Exception("Dataset is Empty")

            missing_values = df.isnull().sum().sum()

            logger.info(f"Missing Values : {missing_values}")

            duplicate_rows = df.duplicated().sum()

            logger.info(f"Duplicate Rows : {duplicate_rows}")

            logger.info(f"Dataset Shape : {df.shape}")

            logger.info(f"\nData Types\n{df.dtypes}")

            logger.info("Data Validation Completed Successfully")

            return True

        except Exception as e:

            logger.error("Error During Data Validation")

            raise CustomException(e, sys)