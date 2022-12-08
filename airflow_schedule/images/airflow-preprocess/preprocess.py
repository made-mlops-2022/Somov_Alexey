import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("preprocess")
@click.option("--input-data-dir")
@click.option("--output-data-dir")
@click.option("--mode")
def preprocess(input_data_dir: str, output_data_dir: str, mode: str):
    data = pd.read_csv(os.path.join(input_data_dir, "data.csv"), index_col=0)

    os.makedirs(output_data_dir, exist_ok=True)
    if mode == 'train':
        _TARGET_INPUT_PATH = os.path.join(input_data_dir, "target.csv")
        _TRAIN_DATA_PATH = os.path.join(output_data_dir, "train.csv")
        _VALID_DATA_PATH = os.path.join(output_data_dir, "valid.csv")

        target = pd.read_csv(_TARGET_INPUT_PATH, index_col=0)
        data = data.join(target, how='left')
        data = clean_dataset(data, target_col='target')
        train_data, valid_data = train_test_split(data, test_size=0.4, random_state=42, shuffle=True)
        train_data.to_csv(_TRAIN_DATA_PATH, index=False)
        valid_data.to_csv(_VALID_DATA_PATH, index=False)
    elif mode == 'predict':
        _OUTPUT_DATA_PATH = os.path.join(output_data_dir, "data.csv")

        data = clean_dataset(data)
        data.to_csv(_OUTPUT_DATA_PATH, index=False)
    else:
        raise ValueError(f"Unknown preprocess mode {mode}")


def clean_dataset(data_df: pd.DataFrame, target_col=None):
    if target_col:
        target_na_count = data_df[target_col].isna().sum()
        if target_na_count != 0:
            print(f"WARNING: There is {target_na_count} NaNs in {target_col} column")
            data_df = data_df.dropna(subset=target_col)

    data_df = data_df.dropna(subset=['alcohol', 'proline'])
    data_df = data_df.fillna(-999)
    return data_df



if __name__ == '__main__':
    preprocess()
