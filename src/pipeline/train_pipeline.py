print("Train Pipeline Started")
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.logger import logger


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):

        logger.info("========== Training Pipeline Started ==========")

        # Step 1 - Data Ingestion
        ingestion = DataIngestion()

        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2 - Data Validation
        validation = DataValidation()

        validation.validate_dataset(train_path)
        validation.validate_dataset(test_path)

        # Step 3 - Data Transformation
        transformation = DataTransformation()

        train_array, test_array, _ = transformation.initiate_data_transformation(
            train_path,
            test_path
        )

        # Step 4 - Model Training
        trainer = ModelTrainer()

        model_name, score = trainer.initiate_model_trainer(
            train_array,
            test_array
        )

        # Step 5 - Model Evaluation
        evaluator = ModelEvaluation()

        metrics = evaluator.initiate_model_evaluation(
            test_array
        )

        logger.info("========== Training Pipeline Completed ==========")

        print("\nTraining Completed Successfully")
        print(f"Best Model : {model_name}")
        print(f"R2 Score   : {score}")
        print(metrics)


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.run_pipeline()