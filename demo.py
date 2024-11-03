from hate_text_classifier.logger import logging
from hate_text_classifier.exception import CustomException
from hate_text_classifier.configuration.gcloud_syncer import GcloudSyncer
import sys

# logging.info("Welcome to project")

# try:
#     a=1/0
# except Exception as e:
#     raise CustomException(e,sys)

obj = GcloudSyncer()

obj.sync_folder_from_gcloud("hate_speech_101", "dataset.zip", "dataset.zip")