import sys
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from hate_text_classifier.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from hate_text_classifier.components.data_ingestion import DataIngestion
from hate_text_classifier.components.data_validation import DataValidation
from hate_text_classifier.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from hate_text_classifier.components.data_transformation import DataTransformation


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_trasformation_config = DataTransformationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Getting the data from gcloud bucket storage")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from gcloud bucket storage")

            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifacts)
            is_valid = data_validation.validate_dataset()

            if not is_valid:
                raise Exception("Data validation failed")

            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact, data_transformation_config=self.data_trasformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        logging.info("Starting the run pipeline")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)

            logging.info("Run pipeline completed")
        except Exception as e:
            raise CustomException(e, sys)