import os

import click
import pickle as pkl
import pandas as pd


@click.command("predict")
@click.option("--input-data-dir")
@click.option("--prod-model-dir")
@click.option("--predict-data-dir")
def predict(input_data_dir: str, prod_model_dir: str, predict_data_dir: str):
    os.makedirs(predict_data_dir, exist_ok=True)

    data = pd.read_csv(os.path.join(input_data_dir, "data.csv"))
    with open(os.path.join(prod_model_dir, "model.pkl"), 'rb') as fd:
        model = pkl.load(fd)

    y_pred = pd.DataFrame(model.predict(data), columns=['predicted'])
    y_pred.to_csv(os.path.join(predict_data_dir, "predict.csv"))


if __name__ == '__main__':
    predict()
    
