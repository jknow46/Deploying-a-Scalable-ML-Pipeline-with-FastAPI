import pytest
# TODO: add necessary import
import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

DATA_PATH = os.path.join("data", "census.csv")
CAT_FEATURES = [
    "workclass","education","marital-status","occupation",
    "relationship","race","sex","native-country",
]


def _load_sample():
    df = pd.read_csv(DATA_PATH)
    df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    train, test = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df["salary"] if "salary" in df.columns else None
    )
    X_train, y_train, enc, lb = process_data(train, CAT_FEATURES, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, CAT_FEATURES, label="salary", training=False, encoder=enc, lb=lb)
    return X_train, X_test, y_train, y_test


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # process_data returns numpy arrays with matching rows and expected dims
    """
    X_train, X_test, y_train, y_test = _load_sample()
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.ndim == 2
    assert y_train.ndim == 1
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # train_model returns an estimator exposing predict
    """
    X_train, X_test, y_train, y_test = _load_sample()
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict")
    assert callable(getattr(model, "predict"))
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # compute_model_metrics returns floats in [0,1]
    """
    X_train, X_test, y_train, y_test = _load_sample()
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    for v in (precision, recall, fbeta):
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
    pass
