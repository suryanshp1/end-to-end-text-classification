from hate_text_classifier.logger import logging
from hate_text_classifier.exception import CustomException
import sys

# logging.info("Welcome to project")

try:
    a=1/0
except Exception as e:
    raise CustomException(e,sys)