import re
import sys
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from hate_text_classifier.constants import *
from hate_text_classifier.entity.config_entity import DataTransformationConfig
from hate_text_classifier.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
import pandas as pd
import numpy as np

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def imbalance_data_cleaning(self):
        logging.info("Entered imbalanced data cleaning")
        try:
            imbalanced_data = pd.read_csv(self.data_ingestion_artifact.imbalance_data_file_path)
            imbalanced_data.drop(self.data_transformation_config.ID, axis=self.data_transformation_config.AXIS, inplace=self.data_transformation_config.INPLACE)
            logging.info("Exited imbalanced data cleaning")
            return imbalanced_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def raw_data_cleaning(self):
        logging.info("Entered raw data cleaning")
        try:
            raw_data = pd.read_csv(self.data_ingestion_artifact.raw_data_file_path)
            raw_data.drop(self.data_transformation_config.DROP_COLUMNS, axis=self.data_transformation_config.AXIS, inplace=self.data_transformation_config.INPLACE)
            raw_data[raw_data[self.data_transformation_config.CLASS] == 0][self.data_transformation_config.CLASS] = 1

            raw_data[self.data_transformation_config.CLASS].replace({0: 1}, inplace=True)

            raw_data[self.data_transformation_config.CLASS].replace({2: 0}, inplace=True)

            raw_data = raw_data.rename(columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL})

            logging.info("Exited raw data cleaning")

            return raw_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def concat_dataframe(self):
        logging.info("Entered concat dataframe")
        try:
            imbalanced_data = self.imbalance_data_cleaning()
            raw_data = self.raw_data_cleaning()
            df = pd.concat([imbalanced_data, raw_data])
            logging.info("Exited concat dataframe")
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def cancat_data_cleaning(self, words):
        logging.info("Entered cancat data cleaning")
        try:
            stemmer = nltk.SnowballStemmer("english")
            stopword = stopwords.words("english")
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(words) for word in words.split(' ')]
            words=" ".join(words)

            logging.info("Exited cancat data cleaning")

            return words
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Entered initiate_data_transformation")
            df = self.concat_dataframe()
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.cancat_data_cleaning)
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACT_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
            )
            logging.info("Exited initiate_data_transformation")

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)