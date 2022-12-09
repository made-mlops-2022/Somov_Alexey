import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config_classes import SplittingParams

logger = logging.getLogger(__name__)


def read_data(path: str) -> pd.DataFrame:
    logger.info("read data from %s", path)

    data: pd.DataFrane = pd.read_csv(path)

    return data


def split_train_val_data(
        data: pd.DataFrame,
        params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("split dataset with %s", params)

    train_data, val_data = train_test_split(
        data, test_size=params.val_size,
        random_state=params.random_state)

    return train_data, val_data