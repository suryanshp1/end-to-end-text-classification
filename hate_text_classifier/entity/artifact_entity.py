from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    imbalance_data_file_path: str
    raw_data_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_data_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    x_test_path: str
    y_test_path: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool


@dataclass
class ModelPusherArtifact:
    bucket_name: str