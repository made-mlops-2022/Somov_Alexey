import pytest
import numpy as np
import pandas as pd

from src.model import eval_metrics


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (pd.Series([0, 1]), np.array([0.1, 1]), {"acc": 1, "auc": 1, "f1": 1}),
        (pd.Series([0, 1]), np.array([0.1, 0]), {"acc": 0.5, "auc": 0, "f1": 0}),
        (pd.Series([0, 1]), np.array([0.6, 0.3]), {"acc": 0, "auc": 0, "f1": 0}),
    ],
)
def test_eval_metrics(y_true, y_pred, result):
    assert eval_metrics(y_true, y_pred) == result