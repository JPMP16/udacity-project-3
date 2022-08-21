import pytest
import numpy as np
import pickle
import starter.starter.ml.model
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def X():
    X = np.load("starter/data/test_X.npy")
    return X

@pytest.fixture
def y():
    y = np.load("starter/data/test_y.npy")
    return y

@pytest.fixture
def preds():
    preds = np.load("starter/data/test_preds.npy")
    return preds

@pytest.fixture
def rf():
    with open('starter/model/rf.pkl', 'rb') as fid:
        rf = pickle.load(fid)
    return rf


def test_train_model(X, y):
    rf = starter.starter.ml.model.train_model(X,y)
    assert isinstance(rf, RandomForestClassifier)

def test_compute_model_metrics(y, preds):
    precision, recall, fbeta = starter.starter.ml.model.compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

def test_inference(rf, X):
    preds = starter.starter.ml.model.inference(rf, X)
    assert isinstance(preds, np.ndarray)
