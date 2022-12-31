import os
from datetime import timedelta

from airflow.models import Variable


DATA_DIR = "/opt/airflow/data"
LOCAL_DATA_PATH = Variable.get("LOCAL_DATA_PATH")
PRODUCTION_MODEL_DIR = os.path.join(DATA_DIR, "models")

DEFAULT_DAG_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

DOCKER_CONFIG = {
    "network_mode": "bridge",
    "volumes": [f"{LOCAL_DATA_PATH}:{DATA_DIR}"],
    "auto_remove": True,
}