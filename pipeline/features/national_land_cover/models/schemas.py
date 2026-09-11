from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TableFileFormat(str, Enum):
    csv = "CSV"
    geo_json = "GEO_JSON"
    shp = "SHP"


class ClassifierType(str, Enum):
    smile_random_forest = "smileRandomForest"
    smile_cart = "smileCart"
    smile_gradient_tree_boost = "smileGradientTreeBoost"


class TaskResponse(BaseModel):
    task_id: Optional[str] = None
    state: Optional[str] = None
    description: Optional[str] = None
    status: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    gcs_uri: str
    filename: str
    content_type: Optional[str] = None


class IngestSamplesRequest(BaseModel):
    gcs_uri: str
    asset_id: str
    file_format: TableFileFormat
    description: Optional[str] = None


class IngestSamplesResponse(BaseModel):
    task_id: str
    asset_id: str
    operation: Dict[str, Any]


class ExtractEmbeddingsRequest(BaseModel):
    samples_asset_id: str
    label_property: str = "class"
    embedding_asset_id: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    year: int = Field(..., ge=2017, le=2100)
    scale: int = Field(default=10, gt=0)
    output_gcs_bucket: str
    output_gcs_prefix: str
    output_asset_id: str
    include_geometries: bool = True


class ExtractEmbeddingsResponse(BaseModel):
    gcs_export_task: TaskResponse
    ee_export_task: TaskResponse
    output_gcs_uri_prefix: str
    output_asset_id: str


class TrainModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    training_samples_asset_id: str
    label_property: str = "class"
    classifier_type: ClassifierType = ClassifierType.smile_random_forest
    classifier_params: Dict[str, Any] = Field(default_factory=lambda: {"numberOfTrees": 100, "seed": 42})
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    random_seed: int = 42
    input_properties: Optional[List[str]] = None
    model_metadata_asset_id: Optional[str] = None
    validation_asset_id: str
    validation_gcs_bucket: str
    validation_gcs_prefix: str


class ValidateModelRequest(BaseModel):
    validation_samples_asset_id: str
    label_property: str = "class"
    classifier_type: ClassifierType = ClassifierType.smile_random_forest
    classifier_params: Dict[str, Any] = Field(default_factory=lambda: {"numberOfTrees": 100, "seed": 42})
    training_samples_asset_id: Optional[str] = None
    input_properties: Optional[List[str]] = None
    classifier_json: Optional[Dict[str, Any]] = None
    validation_asset_id: str
    validation_gcs_bucket: str
    validation_gcs_prefix: str

    @model_validator(mode="after")
    def require_training_or_serialized_classifier(self) -> "ValidateModelRequest":
        if not self.training_samples_asset_id and not self.classifier_json:
            raise ValueError("Provide either training_samples_asset_id to reconstruct the classifier or classifier_json when supported by Earth Engine.")
        return self


class ValidationMetrics(BaseModel):
    accuracy: Any
    kappa: Any
    producers_accuracy: Any
    consumers_accuracy: Any
    confusion_matrix: Any


class TrainModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    validation_metrics: ValidationMetrics
    validation_gcs_export_task: TaskResponse
    validation_ee_export_task: TaskResponse
    model_metadata_export_task: Optional[TaskResponse] = None


class ValidateModelResponse(BaseModel):
    validation_metrics: ValidationMetrics
    validation_gcs_export_task: TaskResponse
    validation_ee_export_task: TaskResponse


# --- Unified Request Payload ---

class UnifiedLandCoverRequest(BaseModel):
    input_asset_id: str
    cloud_project: str
    run_name: str
    longitude_column: str = "lon"
    latitude_column: str = "lat"
    block_crs: str = "EPSG:3857"
    target_column: str = "LOI_PCT"
    target_threshold: float = 30.0
    reference_year: int = 2022
    block_size_m: int = 10000
    test_block_fraction: float = 0.2
    points_per_block: List[Union[int, str]] = Field(default_factory=lambda: [1, 2, 5, 10, 20, "all"])
    block_fractions: List[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    auc_tolerance: float = 0.01
    number_of_trees: int = 100
    number_of_embedding_bands: int = 64
    sampling_scale_m: int = 10
    seed: int = 42
    asset_root: str
    sample_asset_id: str
    model_asset_id: str
    results_bucket: str
    results_prefix: str
