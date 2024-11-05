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
        self.NEW_DATA_ARTIFACTS_DIR = os.path.join(self.DATA_INGESTION_ARTIFACT_DIR, DATA_INGESTION_RAW_DATA_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACT_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACT_DIR, ZIP_FILE_NAME)

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACT_DIR = os.path.join(os.getcwd(),ARTIFACT_DIR, DATA_TRANSFORMATION_ARTIFACT_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACT_DIR, TRANSFORMED_FILE_NAME)
        self.DATA_DIR = DATA_DIR
        self.ID = ID
        self.LABEL = LABEL
        self.TWEET = TWEET
        self.AXIS = AXIS
        self.INPLACE = INPLACE
        self.CLASS = CLASS
        self.DROP_COLUMNS = DROP_COLUMNS