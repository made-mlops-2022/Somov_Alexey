import os

import click
import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier


@click.command("train")
@click.option("--input-data-dir")
@click.option("--output-model-dir")
def train(input_data_dir: str, output_model_dir: str):
    _TARGET_COL = 'target'

    data = pd.read_csv(os.path.join(input_data_dir, "train.csv"))
    y_train = data[_TARGET_COL]
    X_train = data.drop(_TARGET_COL, axis=1)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(output_model_dir, exist_ok=True)
    with open(os.path.join(output_model_dir, "model.pkl"), 'wb') as fd:
        pkl.dump(model, fd)


if __name__ == '__main__':
    train()

