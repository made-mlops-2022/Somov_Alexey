import os

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator

from dags_config import DEFAULT_DAG_ARGS, DATA_DIR, DOCKER_CONFIG


with DAG(
        dag_id="data_generator",
        schedule_interval="@daily",
        start_date=days_ago(8),
        default_args=DEFAULT_DAG_ARGS,
) as dag:
    SCHEDULE_DATE = "{{ ds }}"
    INPUT_DATA_DIR = os.path.join(DATA_DIR, "raw", SCHEDULE_DATE)

    data_generator = DockerOperator(
        task_id="data_generator",
        image="airflow-download",
        command=f"--output-dir={INPUT_DATA_DIR}",
        **DOCKER_CONFIG,
    )

    data_generator