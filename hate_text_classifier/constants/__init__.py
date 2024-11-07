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

# Data Validation constants
IMBALANCED_DATA_COLUMNS: list = ['id', 'label', 'tweet']
RAW_DATA_COLUMNS: list = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither','class', 'tweet']


# Data Transformation constants
DATA_TRANSFORMATION_ARTIFACT_DIR: str = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME: str = "final.csv"
DATA_DIR: str = "data"
ID = "id"
AXIS = 1
INPLACE = True
DROP_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS = "class"

# Model Trainer constants
MODEL_TRAINER_ARTIFACT_DIR: str = "ModelTrainerArtifacts"
TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_NAME: str = "model.h5"
X_TEST_FILE_NAME: str = "x_test.csv"
Y_TEST_FILE_NAME: str = "y_test.csv"

X_TRAIN_FILE_NAME: str = "x_train.csv"

RANDOM_STATE: int = 42
EPOCH: int = 2
BATCH_SIZE: int = 128
VALIDATION_SPLIT: float = 0.2


# Model Architecture constants
MAX_WORDS: int = 50000
MAX_LEN: int = 300
LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
ACTIVATION = "sigmoid"

# Model Evaluation constants
MODEL_EVALUATION_ARTIFACT_DIR: str = "ModelEvaluationArtifacts"
BEST_MODEL_DIR: str = "best_model"
MODEL_EVALUATION_FILE_NAME: str = "loss.csv"

MODEL_NAME = "model.h5"
APP_HOST = "0.0.0.0"
APP_PORT = 8080