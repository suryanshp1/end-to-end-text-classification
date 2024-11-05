from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    imbalance_data_file_path: str
    raw_data_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_data_path: str