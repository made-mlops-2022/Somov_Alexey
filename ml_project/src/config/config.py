import logging

import yaml
from marshmallow_dataclass import class_schema

from src.config.config_classes import Config

logger = logging.getLogger(__name__)

ConfigSchema = class_schema(Config)


def read_config(config_path: str) -> Config:
    logger.info("building config from %s", config_path)

    with open(config_path) as fd:
        config_dict = yaml.safe_load(fd)

    return build_config_from_dict(config_dict)


def build_config_from_dict(config: dict) -> Config:
    logger.info("building config from dict")

    config_schema = ConfigSchema()
    return config_schema.load(config)
