"""xgboost_quickstart: A Flower / XGBoost app."""

from logging import INFO

import xgboost as xgb
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .label_encoder import MyLabelEncoder
import numpy as np

UNIQUE_LABELS = [0, 1]
FEATURES = ['sex', 'age', 'race', 'marital-status', 'education', 'native-country', 'workclass', 'occupation']
TARGET = 'salary-class'

# def train_test_split(partition, test_fraction, seed):
#     """Split the data into train and validation set given split rate."""
#     train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
#     partition_train = train_test["train"]
#     partition_test = train_test["test"]

#     num_train = len(partition_train)
#     num_test = len(partition_test)

#     return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data[FEATURES]
    y = data[TARGET]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


fds = None  # Cache FederatedDataset
CACHE_ROOT_DIR = 'results/' # Cache anonymized dataset


def load_data(partition_id, num_clients, context):
    # Only initialize `FederatedDataset` once
    anon_method = context.run_config["anon-method"]
    k = int(context.run_config["k"])

    cache_dir = f'{CACHE_ROOT_DIR}/{anon_method}/{k}'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'{CACHE_ROOT_DIR}/{anon_method}/{k}/{partition_id}_anon.csv'
    cache_file_orig = f'{CACHE_ROOT_DIR}/{anon_method}/{k}/{partition_id}_orig.csv'
    cache_file_ncp = f'{CACHE_ROOT_DIR}/{anon_method}/{k}/{partition_id}_ncp.txt'
    cache_file_rc = f'{CACHE_ROOT_DIR}/{anon_method}/{k}/{partition_id}_record_linkage.txt'

    if os.path.exists(cache_file):
        print(f"cache hit: {cache_file}")
        dataset_anon = pd.read_csv(cache_file)
        dataset = pd.read_csv(cache_file_orig)
    else:
        raise Exception

    if anon_method != "gfkmc" and anon_method != "original":
        mean_of_min_max = (dataset['age'].min() + dataset['age'].max()) / 2

        def convert_age_intervals_to_mean(age_str):
            if age_str == '*':
                return mean_of_min_max
            elif '-' in age_str:
                lower, upper = map(float, age_str.split('-'))
                return (lower + upper) / 2
            else:
                return float(age_str)

        dataset_anon['age'] = dataset_anon['age'].apply(convert_age_intervals_to_mean)

    encoder = MyLabelEncoder()
    dataset_anon = encoder.encode(dataset_anon)


    rnd = 42
    np.random.seed(rnd)
    train_data, valid_data = train_test_split(
        dataset_anon, test_size=0.3, random_state=rnd
    )

    num_train = len(train_data)
    num_val = len(valid_data)

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
