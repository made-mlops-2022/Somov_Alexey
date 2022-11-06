import json
import logging
import logging.config
from pathlib import Path
import yaml
import hydra
from hydra.utils import to_absolute_path

from src.config import build_config_from_dict
from src.data import read_data
from src.data import split_train_val_data

from src.features import build_transformer
from src.features import make_features
from src.features import save_transformer
from src.model import get_model
from src.model import eval_metrics
from src.model import save_model


def setup_logger(logger_config_path):
    with open(logger_config_path) as fd:
        logger_config = yaml.safe_load(fd)
    logging.config.dictConfig(logger_config)


def train(train_config):
    data_df = read_data(to_absolute_path(train_config.input_data_path))
    train_df, val_df = split_train_val_data(data_df, train_config.split_params)

    train_target = train_df[train_config.feature_params.target_col]
    train_df = train_df.drop(train_config.feature_params.target_col, axis=1)

    transformer = build_transformer(train_config.feature_params.transform_params)
    transformer.fit(train_df)

    transformer_path = Path(to_absolute_path(train_config.feature_transformer_path))
    transformer_path.parent.mkdir(exist_ok=True, parents=True)
    save_transformer(transformer, transformer_path)

    train_features = make_features(transformer, train_df)

    model = get_model(train_config.train_params)
    model.fit(train_features, train_target)

    save_model(model, to_absolute_path(train_config.output_model_path))

    y_true = val_df[train_config.feature_params.target_col]
    val_df = val_df.drop(train_config.feature_params.target_col, axis=1)
    val_features = make_features(transformer, val_df)

    y_pred = model.predict_proba(val_features)[:, 1]

    metrics = eval_metrics(y_true, y_pred)
    with open(to_absolute_path(train_config.metric_path), "w") as fd:
        json.dump(metrics, fd)


@hydra.main()
def main(config_dict):
    train_config = build_config_from_dict(config_dict)
    setup_logger(to_absolute_path(train_config.logger_config_path))
    train(train_config)


if __name__ == "__main__":
    main()