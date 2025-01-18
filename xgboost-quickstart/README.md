# adult dataset federated learning

Dataset: https://huggingface.co/datasets/scikit-learn/adult-census-income

Code based on: https://github.com/adap/flower/tree/main/examples/xgboost-quickstart

## Usage

You must run `quickstart-sklearn` first to generate the datasets, and manually copy `quickstart-sklearn/results/` to `xgboost-quickstart/results/` before running this.
It works like this to ensure that the anonymized datasets are the same.

### 1. create venv

```{bash}
python -m venv .venv
source .venv/bin/activate
```

### 2. install dependencies

- Python 3.9.19

```{bash}
pip install -e .
```

### 3. run simulation

```{bash}
# ./run.sh <k> <anon_method>
./run.sh 5 mondrian
```
