import sys
from hate_text_classifier.exception import CustomException
from hate_text_classifier.logger import logging
from hate_text_classifier.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from hate_text_classifier.components.data_ingestion import DataIngestion
from hate_text_classifier.components.data_validation import DataValidation
from hate_text_classifier.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from hate_text_classifier.components.data_transformation import DataTransformation
from hate_text_classifier.components.model_trainer import ModelTrainer
from hate_text_classifier.components.model_evaluation import ModelEvaluation
from hate_text_classifier.components.model_pusher import ModelPusher


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_trasformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Getting the data from gcloud bucket storage")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from gcloud bucket storage")

            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifacts)
            is_valid = data_validation.validate_dataset()

            if not is_valid:
                raise Exception("Data validation failed")

            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Started data transformation")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact, data_transformation_config=self.data_trasformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Ended data transformation")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Started model trainer")
            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Ended model trainer")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact) -> ModelEvaluationArtifact:
        try:
            logging.info("Started model evaluation")
            model_evaluation = ModelEvaluation(model_evaluation_config=self.model_evaluation_config, model_trainer_artifact=model_trainer_artifact, data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Ended model evaluation")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_pusher(self):
        try:
            logging.info("Started model pusher")
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Ended model pusher")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        logging.info("Starting the run pipeline")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)

            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact, data_transformation_artifact=data_transformation_artifact)

            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Training model is not better than previous model")

            self.start_model_pusher()
            logging.info("Run pipeline completed")
        except Exception as e:
            raise CustomException(e, sys)