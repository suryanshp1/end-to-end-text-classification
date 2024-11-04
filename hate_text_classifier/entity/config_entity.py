from dataclasses import dataclass
from hate_text_classifier.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.ZIP_FILE_NAME: str = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACT_DIR = os.path.join(os.getcwd(),ARTIFACT_DIR, DATA_INGESTION_ARTIFACT_DIR)
        self.DATA_ARTIFACTS_DIR = os.path.join(self.DATA_INGESTION_ARTIFACT_DIR, DATA_INGESTION_IMBALANCE_DATA_DIR)
        self.DATA_NEW_ARTIFACTS_DIR = os.path.join(self.DATA_INGESTION_ARTIFACT_DIR, DATA_INGESTION_RAW_DATA_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACT_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACT_DIR, ZIP_FILE_NAME)