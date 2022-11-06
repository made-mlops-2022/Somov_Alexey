import pandas as pd

from src.data import read_data, split_train_val_data
from src.config.config_classes import SplittingParams


def test_read_data(tmp_path):
    path = tmp_path / "predicts.csv"
    pd.DataFrame([[1, 2], ["a", "b"]]).to_csv(path, index=False)
    data = read_data(path)

    assert data.shape == (2, 2)


def test_split_train_val_data():
    test_df = pd.DataFrame([[1, 2], ["a", "b"], [1, 2], ["a", "b"]])
    test_params = SplittingParams(0.25, 42)

    train_df, val_df = split_train_val_data(test_df, test_params)
    assert train_df.shape == (3, 2)
    assert val_df.shape == (1, 2)