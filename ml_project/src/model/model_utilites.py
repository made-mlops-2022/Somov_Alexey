import logging
from typing import NoReturn
from typing import Dict
import pickle
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def save_model(model: BaseEstimator, output_path: str) -> NoReturn:
    logger.info("save model %s to %s", model.__class__.__name__, output_path)

    with open(output_path, "wb") as fd:
        pickle.dump(model, fd)


def load_model(model_path: str) -> BaseEstimator:
    logger.info("load model from %s", model_path)

    with open(model_path, "rb") as fd:
        model = pickle.load(fd)

    return model


def eval_metrics(y_true: pd.Series, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    logger.info("evaluating model on dataset %s with threshold %s", y_pred.shape, threshold)

    metrics = {
        "acc": accuracy_score(y_true, y_pred > threshold),
        "auc": roc_auc_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred > threshold),
    }
    logger.info("metrics %s", metrics)

    return metrics
