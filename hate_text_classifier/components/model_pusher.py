import sys
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from hate_text_classifier.entity.artifact_entity import ModelPusherArtifact
from hate_text_classifier.entity.config_entity import ModelPusherConfig
from hate_text_classifier.configuration.gcloud_syncer import GcloudSyncer

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
            self.model_pusher_config = model_pusher_config
            self.gcloud = GcloudSyncer()

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            self.gcloud.sync_folder_to_gcloud(self.model_pusher_config.BUCKET_NAME, self.model_pusher_config.TRAINED_MODEL_PATH, self.model_pusher_config.MODEL_NAME)

            logging.info("Uploaded best model to google clod storage")
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.BUCKET_NAME)
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e,sys)