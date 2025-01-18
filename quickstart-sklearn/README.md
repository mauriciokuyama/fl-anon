# adult dataset federated learning

dataset: https://huggingface.co/datasets/scikit-learn/adult-census-income

code based on: https://github.com/adap/flower/tree/main/examples/quickstart-sklearn-tabular

## Usage

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
