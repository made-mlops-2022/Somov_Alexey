import logging

from src.config.config_classes import TrainingParams
import src.model.models as M
from .model_utilites import eval_metrics, save_model, load_model

logger = logging.getLogger(__name__)


def get_model(params: TrainingParams):
    logger.debug("get model %s with params %s", params.model_type, params.params)

    model = getattr(M, params.model_type)(**params.params)

    return model


__all__ = ["get_model", "eval_metrics", "save_model", "load_model"]
