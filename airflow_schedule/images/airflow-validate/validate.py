import os

import json
import click
import pickle as pkl
import pandas as pd

from sklearn import metrics


@click.command("validate")
@click.option("--input-data-dir")
@click.option("--input-model-dir")
@click.option("--prod-model-dir")
def validate(input_data_dir: str, input_model_dir, prod_model_dir: str):
    _TARGET_COL = 'target'
    _TRAIN_DATA_PATH = os.path.join(input_data_dir, "train.csv")
    _VALID_DATA_PATH = os.path.join(input_data_dir, "valid.csv")
    _PROD_METRICS_PATH = os.path.join(prod_model_dir, "metrics.json")
    _PROD_MODEL_PATH = os.path.join(prod_model_dir, "model.pkl")

    train_data = pd.read_csv(_TRAIN_DATA_PATH)
    valid_data = pd.read_csv(_VALID_DATA_PATH)
    with open(os.path.join(input_model_dir, "model.pkl"), 'rb') as fd:
        model = pkl.load(fd)

    train_metrics = calculate_metrics(model, train_data, target_col=_TARGET_COL)
    save_metrics(os.path.join(input_model_dir, "train_metrics.json"), train_metrics)

    valid_metrics = calculate_metrics(model, valid_data, target_col=_TARGET_COL)
    prod_metrics = read_metrics(_PROD_METRICS_PATH)
    if (valid_metrics['roc_auc'] > prod_metrics['roc_auc']
            and valid_metrics['f1_score'] > prod_metrics['f1_score']):
        save_metrics(_PROD_METRICS_PATH, valid_metrics)
        with open(_PROD_MODEL_PATH, 'wb') as fd:
            pkl.dump(model, fd)


def calculate_metrics(model, data_df, target_col='target'):
    X_val = data_df.drop(target_col, axis=1)
    y_true = data_df[target_col]
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val)

    roc_auc = metrics.roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    f1_score = metrics.f1_score(y_true, y_pred, average='macro')
    return {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def save_metrics(metrics_path, cur_metrics):
    with open(metrics_path, 'w') as fd:
        json.dump(cur_metrics, fd)


def read_metrics(metrics_path):
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as fd:
            prod_metrics = json.load(fd)
    else:
        prod_metrics = {
            "roc_auc": 0.0,
            "f1_score": 0.0,
        }
    return prod_metrics


if __name__ == '__main__':
    validate()
    
    
