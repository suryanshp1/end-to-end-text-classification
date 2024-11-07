import os
import sys
import pandas as pd
import pickle
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from sklearn.model_selection import train_test_split
from hate_text_classifier.constants import *
from hate_text_classifier.entity.config_entity import ModelTrainerConfig
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate_text_classifier.ml.model import ModelArchitecture
from hate_text_classifier.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def splitting_data(self, csv_file_path):
        try:
            logging.info(f"Entered into splitting data function")
            df = pd.read_csv(csv_file_path, index_col=False)
            logging.info(f"Data loaded successfully")

            df[TWEET]=df[TWEET].astype(str)

            x=df[TWEET]
            y=df[LABEL]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)
            logging.info(f"Data splitted successfully")

            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
        
    def tokenizing(self, x_train):
        try:
            logging.info(f"Entered into tokenizing function")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            
            tokenizer.fit_on_texts(x_train)

            sequences = tokenizer.texts_to_sequences(x_train)
            
            sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)

            logging.info(f"tokenizing done successfully")

            return sequences_matrix,tokenizer

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Entered into initiate_model_trainer function")
            x_train, x_test, y_train, y_test = self.splitting_data(csv_file_path=self.data_transformation_artifact.transformed_data_path)

            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            sequence_matrix,tokenizer =self.tokenizing(x_train)

            model.fit(sequence_matrix,y_train,batch_size=self.model_trainer_config.BATCH_SIZE,epochs = self.model_trainer_config.EPOCH,validation_split=self.model_trainer_config.VALIDATION_SPLIT)

            with open("tokenizer.pickle", "wb") as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)

            logging.info("Saving trained model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
            )

            logging.info("returning the model trainer artifact")
            return model_trainer_artifact


        except Exception as e:
            raise CustomException(e, sys)