import logging
from typing import List

import requests
import hydra
from hydra.utils import to_absolute_path

from online_inference.config.config_classes import QueryData

logger = logging.getLogger(__name__)


def read_queries(path: str) -> List[QueryData]:
    logger.info('read queries from %s', path)
    with open(path, 'r') as fd:
        data = fd.read().splitlines()
    columns = data[0].split(',')
    queries = [QueryData(**dict(zip(columns, row.split(',')))) for row in data[1:]]
    return queries


@hydra.main(config_path="..")
def main(request_config):
    queries = read_queries(to_absolute_path(request_config.request_data_path))
    app_address = f"http://{request_config.app_host}:{request_config.app_port}"
    logger.info('start sending queries to %s/predict', app_address)
    for query in queries:
        logger.info("send requests %s", query)
        response = requests.post(f"{app_address}/predict", data=query.json())
        logger.info("response %s", response.json())


if __name__ == "__main__":
    main()