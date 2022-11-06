import pandas as pd
from sklearn.utils.validation import check_is_fitted
from hydra.utils import to_absolute_path

from train import train
from predict import predict
from src.config import read_config
from src.features import load_transformer
from src.model import load_model


def test_intergated(fake_dataset, config_path):
    train_config = read_config(config_path)
    train(train_config)
    transformer = load_transformer(to_absolute_path(train_config.feature_transformer_path))
    model = load_model(to_absolute_path(train_config.output_model_path))

    check_is_fitted(transformer)
    check_is_fitted(model)

    predict(train_config)

    y_pred = pd.read_csv(to_absolute_path(train_config.predict_path))
    assert y_pred.shape[0] == len(fake_dataset.splitlines()) - 1
    assert y_pred.values.max() < 1.0
    assert y_pred.values.min() > 0.0