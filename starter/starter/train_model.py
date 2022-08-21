# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import load_data, process_data
from ml.model import train_model, compute_model_metrics, inference, save_model_encoder

if __name__ == "__main__":
    # Read data
    data = load_data('../data/census.csv')
    # Split train and test datasets
    train, test = train_test_split(data, test_size=0.20)
    # Set categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Process train and test datasets
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # Train Model
    model = train_model(X_train, y_train)
    # Predict using model
    preds = inference(model, X_test)
    # Model results
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(precision, recall, fbeta)
    # Save artifacts
    save_model_encoder(model, encoder, lb)
