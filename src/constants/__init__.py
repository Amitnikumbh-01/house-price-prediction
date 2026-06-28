import os

# Project Root Directory
ROOT_DIR = os.getcwd()

# Data Folder
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Artifacts Folder
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Dataset File
DATA_FILE_NAME = "Housing.csv"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

# Train-Test Files
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

TRAIN_FILE_PATH = os.path.join(ARTIFACTS_DIR, TRAIN_FILE_NAME)
TEST_FILE_PATH = os.path.join(ARTIFACTS_DIR, TEST_FILE_NAME)

# Model File
MODEL_FILE_NAME = "model.pkl"
MODEL_FILE_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILE_NAME)

# Preprocessor File
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"
PREPROCESSOR_FILE_PATH = os.path.join(
    ARTIFACTS_DIR,
    PREPROCESSOR_FILE_NAME
)