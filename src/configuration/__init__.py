import os

from src.constants import (
    ARTIFACTS_DIR,
    DATA_FILE_PATH,
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
)

from src.entity import DataIngestionConfig


class ConfigurationManager:

    def __init__(self):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def get_data_ingestion_config(self):

        return DataIngestionConfig(
            raw_data_path=DATA_FILE_PATH,
            train_data_path=TRAIN_FILE_PATH,
            test_data_path=TEST_FILE_PATH
        )