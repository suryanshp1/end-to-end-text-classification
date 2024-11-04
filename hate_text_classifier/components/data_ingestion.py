import os
import sys
from zipfile import ZipFile
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from hate_text_classifier.entity.config_entity import DataIngestionConfig
from hate_text_classifier.entity.artifact_entity import DataIngestionArtifact
from hate_text_classifier.configuration.gcloud_syncer import GcloudSyncer


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GcloudSyncer()

    def get_data_from_gcloud(self) -> None:
        try:
            logging.info(
                "Downloading data from gcloud in Data Ingestion component"
            )
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACT_DIR, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(
                gcp_bucket_url=self.data_ingestion_config.BUCKET_NAME,
                filename=self.data_ingestion_config.ZIP_FILE_NAME,
                destination=self.data_ingestion_config.DATA_INGESTION_ARTIFACT_DIR,
            )
            logging.info(
                "Downloaded data from gcloud in Data Ingestion component successfully !!!"
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def unzip_and_clean(self):
        logging.info("Unzipping data")
        try:
            with ZipFile(
                self.data_ingestion_config.ZIP_FILE_PATH, "r"
            ) as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            
            logging.info("Unzipped data")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Initiating data ingestion")
        try:
            self.get_data_from_gcloud()
            logging.info("Fetched data from gcloud")
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped file and aplit into train and valid")

            data_ingestion_artifact = DataIngestionArtifact(
                imbalance_data_file_path=imbalance_data_file_path,
                raw_data_file_path=raw_data_file_path
            )

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
