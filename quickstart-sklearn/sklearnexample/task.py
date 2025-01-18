# task.py

import numpy as np
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .label_encoder import MyLabelEncoder
from flwr.common import Context
import pandas as pd
import os
import sys
from sklearn import svm
from .record_linkage import record_linkage

UNIQUE_LABELS = [0, 1]
FEATURES = ['sex', 'age', 'race', 'marital-status', 'education', 'native-country', 'workclass', 'occupation']
TARGET = 'salary-class'

def get_model_parameters(model) -> NDArrays:
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def create_log_reg_and_instantiate_parameters(penalty):
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
        solver="saga",
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=len(FEATURES), n_classes=len(UNIQUE_LABELS))
    return model


fds = None  # Cache FederatedDataset
CACHE_ROOT_DIR = 'results/' # Cache anonymized dataset

def load_data(partition_id: int, num_partitions: int, context: Context):
    """Load the data for the given partition."""

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
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="scikit-learn/adult-census-income", partitioners={"train": partitioner}
            )

        dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
        dataset.rename(columns={
            'age': 'age',
            'workclass': 'workclass',
            'fnlwgt': 'final-weight',
            'education': 'education',
            'education.num': 'education-num',
            'marital.status': 'marital-status',
            'occupation': 'occupation',
            'relationship': 'relationship',
            'race': 'race',
            'sex': 'sex',
            'capital.gain': 'capital-gain',
            'capital.loss': 'capital-loss',
            'hours.per.week': 'hours-per-week',
            'native.country': 'native-country',
            'income': 'salary-class'
        }, inplace=True)

        # drop rows with empty fields
        dataset = dataset[FEATURES + [TARGET]]
        dataset.replace('?', np.nan, inplace=True)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)



        if anon_method == "gfkmc":
            from sklearnexample.anon_methods.gfkmc import gfkmc
            from sklearnexample.anon_methods.tree.gentree import read_tree

            categoricals = ['sex', 'race', 'workclass', 'marital-status', 'occupation', 'native-country', 'education']
            numericals = ['age']
            # numericals = []
            sensitives = [TARGET]

            dataset_orig = dataset.copy()
            att_names = dataset[categoricals + sensitives].columns
            att_tree = read_tree('sklearnexample/anon_methods/tree/adult/', att_names)
            table = gfkmc.GFKMCTable(dataset, dataset_orig, numericals, categoricals, sensitives, att_tree)
            remaining_groups = table.initial_clustering_phase(k)
            beta = int(len(remaining_groups) * 0.05)
            table.weighting_phase(beta, remaining_groups)
            table.grouping_phase(k)
            table.adjustment_phase()
            gen_method = 'cluster_centroid'
            dataset_anon = table.cluster_generalization(gen_method)
            ncp = table.ncp(dataset_anon)
            with open(cache_file_ncp, 'w') as f:
                f.write(f'{ncp * 100:.2f}\n')
            matches = record_linkage(dataset, dataset_anon, att_tree, numericals, categoricals)
            with open(cache_file_rc, "w", encoding="utf-8") as f:
                f.write(f'{matches}')

        elif anon_method == "original":
            dataset_anon = dataset
        else:
            import subprocess
            import sys
            import shutil
            from sklearnexample.anon_methods.tree.gentree import read_tree

            categoricals = ['sex', 'race', 'workclass', 'marital-status', 'occupation', 'native-country', 'education']
            numericals = ['age']
            # numericals = []
            sensitives = [TARGET]
            att_names = dataset[categoricals + sensitives].columns

            python_executable = sys.executable
            dataset_path = f'sklearnexample/anon_methods/k_anon_ml/.datasets/{anon_method}/{k}/{partition_id}/'
            os.makedirs(dataset_path, exist_ok=True)
            dataset.to_csv(f"{dataset_path}/adult.csv", index=True, index_label='ID', sep=';')
            command = f"cd sklearnexample/anon_methods/k_anon_ml && {python_executable} baseline_with_repetitions.py --dataset-path .datasets/{anon_method}/{k}/{partition_id} --results-path .results/{anon_method}/{k}/{partition_id} --start-k {k} --stop-k {k} adult svm {anon_method}; cd -"
            result = subprocess.run(command, shell=True, check=True, text=True)
            dataset_anon = pd.read_csv(f'sklearnexample/anon_methods/k_anon_ml/.results/{anon_method}/{k}/{partition_id}/adult.csv')
            shutil.copy2(f'sklearnexample/anon_methods/k_anon_ml/.results/{anon_method}/{k}/{partition_id}/ncp.txt', cache_file_ncp)
            att_tree = read_tree('sklearnexample/anon_methods/tree/adult/', att_names)
            matches = record_linkage(dataset, dataset_anon, att_tree, numericals, categoricals, num_intervals=True)
            with open(cache_file_rc, "w", encoding="utf-8") as f:
                f.write(f'{matches}')

        dataset_anon.to_csv(cache_file, index=False)
        dataset.to_csv(cache_file_orig, index=False)

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

        # Apply the function to the 'age' column
        dataset_anon['age'] = dataset_anon['age'].apply(convert_age_intervals_to_mean)

    encoder = MyLabelEncoder()
    dataset_anon = encoder.encode(dataset_anon)

    X = dataset_anon[FEATURES]
    y = dataset_anon[TARGET]

    # non_numerical_cols = dataset_anon.select_dtypes(include=['object']).columns
    # dataset_anon_dummies = pd.get_dummies(dataset_anon, columns=non_numerical_cols, drop_first=True)
    # dataset_anon_dummies[TARGET] = pd.factorize(dataset_anon[TARGET])[0]

    # X = dataset_anon_dummies.drop(columns=[TARGET])  # Features
    # y = dataset_anon_dummies[TARGET]

    # Split the on-edge data: 80% train, 20% test
    rnd = 42
    np.random.seed(rnd)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd)
    # X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    # y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
    return X_train, y_train, X_test, y_test
