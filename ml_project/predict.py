import logging
import logging.config

import hydra
import yaml
import pandas as pd
from hydra.utils import to_absolute_path

from src.features import load_transformer
from src.features import make_features
from src.model import load_model
from src.config import build_config_from_dict
from src.data import read_data


def setup_logger(logger_config_path):
    with open(logger_config_path) as fd:
        logger_config = yaml.safe_load(fd)
    logging.config.dictConfig(logger_config)


def predict(predict_config):
    test_df = read_data(to_absolute_path(predict_config.test_data_path))
    test_df = test_df.drop(predict_config.feature_params.target_col, axis=1)

    model_path = to_absolute_path(predict_config.output_model_path)
    model = load_model(model_path)

    transformer = load_transformer(to_absolute_path(predict_config.feature_transformer_path))
    test_features = make_features(transformer, test_df)
    y_pred = pd.DataFrame(model.predict_proba(test_features)[:, 1], columns=["condition"])

    y_pred.to_csv(to_absolute_path(predict_config.predict_path), index=False)


@hydra.main()
def main(config_dict):
    predict_config = build_config_from_dict(config_dict)
    setup_logger(to_absolute_path(predict_config.logger_config_path))
    predict(predict_config)


if __name__ == "__main__":
    main()