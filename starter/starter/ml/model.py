from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pickle


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model: sklearn.ensemble.RandomForestClassifier
        Trained RF machine learning model.
    """
    model = RandomForestClassifier(n_estimators=20, min_samples_split=5)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model_encoder(model, encoder, lb):
    """
    Saves model and encoders used in the training of the model

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.???
        Trained sklearn Encoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.

    Returns
    -------
    None
    """
    with open("../model/rf.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open("../model/encoder.pkl", 'wb') as f:
        pickle.dump(encoder, f)
    with open("../model/lb.pkl", 'wb') as f:
        pickle.dump(lb, f)
