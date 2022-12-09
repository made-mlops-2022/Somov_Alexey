from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional


@dataclass
class SplittingParams:
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class TrainingParams:
    params: dict
    model_type: str = "LogisticRegression"


@dataclass()
class FeatureTransformer:
    trans_name: str
    trans_class: str
    column_names: List[str]
    params: dict = field(default_factory=dict)


@dataclass
class FeatureParams:
    cat_features: Optional[List[str]]
    num_features: Optional[List[str]]
    features_to_drop: Optional[List[str]]
    transform_params: Optional[List[FeatureTransformer]]
    target_col: Optional[str]


@dataclass
class Config:
    logger_config_path: str
    input_data_path: str
    test_data_path: str
    predict_path: str
    output_model_path: str
    metric_path: str
    feature_transformer_path: str
    split_params: SplittingParams
    train_params: TrainingParams
    feature_params: FeatureParams
