import os
import io
import sys
import keras
import pickle
from PIL import Image
from hate_text_classifier.logger import logging
from hate_text_classifier.constants import *
from hate_text_classifier.exception import CustomException
from keras.utils import pad_sequences
from hate_text_classifier.configuration.gcloud_syncer import GcloudSyncer
from hate_text_classifier.entity.config_entity import DataTransformationConfig
from hate_text_classifier.entity.artifact_entity import DataIngestionArtifact
from hate_text_classifier.components.data_transformation import DataTransformation

class PredictionPipeline:
    def __init__(self) -> None:
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.gcloud = GcloudSyncer()
        self.data_transformation = DataTransformation(data_transformation_config=DataTransformationConfig, data_ingestion_artifact=DataIngestionArtifact)

    def get_model_from_gcloud(self):
        logging.info("Entered into get_model_from_gcloud function")
        try:
            os.makedirs(self.model_path, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(
                gcp_bucket_url=self.bucket_name,
                filename=self.model_name,
                destination=self.model_path
            )
            best_model_path = os.path.join(self.model_path, self.model_name)
            logging.info("Exited from get_model_from_gcloud function")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, text):
        try:
            best_model_path: str = self.get_model_from_gcloud()
            load_model = keras.models.load_model(best_model_path)
            with open ("tokenizer.pickle", "rb") as handle:
                tokenizer = pickle.load(handle)

            text = self.data_transformation.cancat_data_cleaning(text)
            text = [text]
            seq = tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            pred = load_model.predict(padded)

            if pred >= 0.5:
                return "Hate and abusive"
            else:
                return "No Hate"
            
        except Exception as e:
            raise CustomException(e, sys)