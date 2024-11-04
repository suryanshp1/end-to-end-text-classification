import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# common constants
TIMESTAMP: str = datetime.now().strftime('%Y%m%d-%H%M%S')
ARTIFACT_DIR: str = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME: str = os.getenv("GCP_BUCKET_NAME")
ZIP_FILE_NAME: str = "dataset.zip"
LABEL: str ="label"
TWEET: str ="tweet"


# Data Ingestion constants
DATA_INGESTION_ARTIFACT_DIR: str = "DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR: str = "imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR: str = "raw_data.csv"