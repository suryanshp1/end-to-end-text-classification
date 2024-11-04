import sys
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from hate_text_classifier.entity.config_entity import DataIngestionConfig
from hate_text_classifier.components.data_ingestion import DataIngestion
from hate_text_classifier.entity.artifact_entity import DataIngestionArtifact


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Getting the data from gcloud bucket storage")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from gcloud bucket storage")

            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        logging.info("Starting the run pipeline")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            logging.info("Run pipeline completed")
        except Exception as e:
            raise CustomException(e, sys)