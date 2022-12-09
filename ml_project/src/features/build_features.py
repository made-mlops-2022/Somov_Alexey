import logging
from typing import List
from typing import NoReturn

import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config.config_classes import FeatureTransformer
import src.features.transformers as T

logger = logging.getLogger(__name__)


def build_transformer(transform_params: List[FeatureTransformer]) -> ColumnTransformer:
    transformers = []
    for param in transform_params:
        logger.info("creating  '%s' column transformers", param.trans_name)
        pipeline = Pipeline([(param.trans_class, getattr(T, param.trans_class)(**param.params)), ])
        transformers.append((param.trans_name, pipeline, param.column_names))
    return ColumnTransformer(transformers)


def save_transformer(transformer: ColumnTransformer, output_path: str) -> NoReturn:
    logger.info("save column transformer %s to %s", transformer.__class__.__name__, output_path)

    with open(output_path, "wb") as fd:
        pickle.dump(transformer, fd)


def load_transformer(transformer_path: str) -> ColumnTransformer:
    logger.info("load column transformer from %s", transformer_path)

    with open(transformer_path, "rb") as fd:
        transformer = pickle.load(fd)

    return transformer


def make_features(transformer: ColumnTransformer, data_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("create features for dataset with shape %s", data_df.shape)

    features = transformer.transform(data_df)
    features_df = pd.DataFrame(features)
    logger.info("created %s new features", features.shape[1])

    return features_df
