import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.configuration import ConfigurationManager


class DataIngestion:

    def __init__(self):

        self.config = ConfigurationManager().get_data_ingestion_config()

    def initiate_data_ingestion(self):

        logger.info("Starting Data Ingestion")

        try:

            df = pd.read_csv(self.config.raw_data_path)

            logger.info("Dataset Loaded Successfully")

            os.makedirs(
                os.path.dirname(self.config.train_data_path),
                exist_ok=True
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.config.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.config.test_data_path,
                index=False
            )

            logger.info("Train and Test datasets saved successfully")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:

            logger.error("Error occurred during Data Ingestion")

            raise CustomException(e, sys)