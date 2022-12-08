import os

from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from dags_config import DATA_DIR, PRODUCTION_MODEL_DIR, DEFAULT_DAG_ARGS, DOCKER_CONFIG


def weekly_train_model(**kwargs):
    day_of_year = kwargs['execution_date'].timetuple().tm_yday
    _MODEL_PATH = os.path.join(PRODUCTION_MODEL_DIR, "model.pkl")
    print(_MODEL_PATH, os.path.exists(_MODEL_PATH), os.getcwd())
    if day_of_year % 7 == 0 or not os.path.exists(_MODEL_PATH):
        return "preprocess_data_train"
    else:
        return "preprocess_data_predict"


with DAG(
        dag_id="daily_pipeline",
        start_date=days_ago(8),
        schedule_interval="@daily",
        default_args=DEFAULT_DAG_ARGS,
) as dag:
    SCHEDULE_DATE = "{{ ds }}"
    INPUT_DATA_DIR = os.path.join(DATA_DIR, "raw", SCHEDULE_DATE)
    PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocessed", SCHEDULE_DATE)
    PREDICT_DATA_DIR = os.path.join(DATA_DIR, "predict", SCHEDULE_DATE)
    WEEKLY_MODEL_DIR = os.path.join(PRODUCTION_MODEL_DIR, SCHEDULE_DATE)

    wait_daily_data = FileSensor(
        task_id="wait_daily_data",
        filepath=f"{INPUT_DATA_DIR}/data.csv",
        poke_interval=30
    )

    weekly_train_branch = BranchPythonOperator(
        task_id='weekly_train_branch',
        python_callable=weekly_train_model,
        provide_context=True,
    )

    preprocess_data_train = DockerOperator(
        task_id="preprocess_data_train",
        image="airflow-preprocess",
        command=f"--input-data-dir={INPUT_DATA_DIR} "
                f"--output-data-dir={PREPROCESSED_DATA_DIR} "
                "--mode=train ",
        **DOCKER_CONFIG,
    )

    model_train = DockerOperator(
        task_id="model_train",
        image="airflow-train",
        command=f"--input-data-dir={PREPROCESSED_DATA_DIR} "
                f"--output-model-dir={WEEKLY_MODEL_DIR} ",
        **DOCKER_CONFIG,
    )

    model_validate = DockerOperator(
        task_id="model_validate",
        image="airflow-validate",
        command=f"--input-data-dir={PREPROCESSED_DATA_DIR} "
                f"--input-model-dir={WEEKLY_MODEL_DIR} "
                f"--prod-model-dir={PRODUCTION_MODEL_DIR} ",
        **DOCKER_CONFIG,
    )

    preprocess_data_predict = DockerOperator(
        task_id="preprocess_data_predict",
        image="airflow-preprocess",
        command=f"--input-data-dir={INPUT_DATA_DIR} "
                f"--output-data-dir={PREPROCESSED_DATA_DIR} "
                "--mode=predict",
        trigger_rule='none_failed',
        **DOCKER_CONFIG,
    )

    model_predict = DockerOperator(
        task_id="model_predict",
        image="airflow-predict",
        command=f"--input-data-dir={PREPROCESSED_DATA_DIR} "
                f"--prod-model-dir={PRODUCTION_MODEL_DIR} "
                f"--predict-data-dir={PREDICT_DATA_DIR} ",
        **DOCKER_CONFIG,
    )

    wait_daily_data >> weekly_train_branch

    weekly_train_branch >> preprocess_data_train >> model_train >> model_validate >> preprocess_data_predict
    weekly_train_branch >> preprocess_data_predict

    preprocess_data_predict >> model_predict