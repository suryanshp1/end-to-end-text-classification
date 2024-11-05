import sys
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from hate_text_classifier.entity.artifact_entity import DataIngestionArtifact
from hate_text_classifier.constants import IMBALANCED_DATA_COLUMNS, RAW_DATA_COLUMNS
import pandas as pd


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
        self.data_ingestion_artifact = data_ingestion_artifact

    def validate_dataset(self) -> bool:
        try:
            logging.info("Validating dataset")
            imbalanced_data = pd.read_csv(
                self.data_ingestion_artifact.imbalance_data_file_path)
            raw_data = pd.read_csv(self.data_ingestion_artifact.raw_data_file_path)

            if len(imbalanced_data.columns) != len(IMBALANCED_DATA_COLUMNS) or len(raw_data.columns) != len(RAW_DATA_COLUMNS):
                return False

            return True
        except Exception as e:
            raise CustomException(e, sys)