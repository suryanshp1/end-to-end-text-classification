import os
import sys
import pandas as pd
import numpy as np
import pickle
import keras
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from sklearn.model_selection import train_test_split
from hate_text_classifier.constants import *
from hate_text_classifier.entity.config_entity import ModelEvaluationConfig
from hate_text_classifier.configuration.gcloud_syncer import GcloudSyncer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate_text_classifier.ml.model import ModelArchitecture
from sklearn.metrics import confusion_matrix
from hate_text_classifier.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataTransformationArtifact


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
        self.model_eval_config = model_eval_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.gcloud = GcloudSyncer()

    def get_best_model_from_gcloud(self) -> str:
        try:
           logging.info("Entered into get_best_model_from_gcloud function")

           os.makedirs(self.model_eval_config.BEST_MODEL_DIR_PATH, exist_ok=True)

           self.gcloud.sync_folder_from_gcloud(
               bucket_name=self.model_eval_config.BUCKET_NAME,
               zip_file_path=self.model_eval_config.MODEL_NAME,
               destination=self.model_eval_config.BEST_MODEL_DIR_PATH
           )

           best_model_path = os.path.join(self.model_eval_config.BEST_MODEL_DIR_PATH, self.model_eval_config.MODEL_NAME)

           logging.info("Exited from get_best_model_from_gcloud function")
           
           return best_model_path
        except Exception as e:
            raise CustomException(e, sys)
        

    def evaluate(self):
        try:
            logging.info("Entered into evaluate function")

            x_test = pd.read_csv(self.model_trainer_artifact.x_test_path, index_col=0)

            y_test = pd.read_csv(self.model_trainer_artifact.y_test_path, index_col=0)

            with open("tokenizer.pickle", "rb") as handle:
                tokenizer = pickle.load(handle)

            load_model = keras.models.load_model(self.model_trainer_artifact.trained_model_path)

            x_test = x_test[TWEET].astype(str)
            
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            accuracy = load_model.evaluate(test_sequences_matrix, y_test)

            logging.info(f"Accuracy: {accuracy}")

            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = []

            for i in lstm_prediction:
                if i[0] >= 0.5:
                    res.append(1)
                else:
                    res.append(0)

            logging.info(f"Confusion Matrix: {confusion_matrix(y_test, res)}")

            logging.info("Exited from evaluate function")

            return accuracy
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Entered into initiate_model_evaluation function")

            trained_model = keras.models.load_model(self.model_trainer_artifact.trained_model_path)
            with open("tokenizer.pickle", "rb") as handle:
                tokenizer = pickle.load(handle)

            trained_model_accuracy = self.evaluate()

            logging.info(f"Feach best model from gcloud bucket storage")

            best_model_path = self.get_best_model_from_gcloud()

            logging.info(f"Chec if the best model is present ot not")

            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("gcloud model is false and currently trained model is true")
            else:
                logging.info("load best modelfetched from gcloud bucket storage")
                best_model = keras.models.load_model(best_model_path)
                best_model_accuracy = self.evaluate()

                logging.info("Comparing loss between best model and trained model")

                if best_model_accuracy > trained_model_accuracy:
                    is_model_accepted = True
                    logging.info("Trained model is not accepted")
                else:
                    is_model_accepted = False
                    logging.info("Trained model is accepted")

            model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=is_model_accepted)
            logging.info(f"returning model evaluation artifact")
            logging.info("Exited from initiate_model_evaluation function")

            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)