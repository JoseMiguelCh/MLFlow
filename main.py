"""
Main script to train a model and log the results with MLFlow
"""
import os
import argparse
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)
EXPERIMENT_NAME = "wine-quality1"
MLFLOW_TRACKING_SERVER_URI = os.environ.get("LOCAL_TRACKING_SERVER_URL")
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_tracking_uri(MLFLOW_TRACKING_SERVER_URI)

def main():
    """
    main function to train a model and log the results with MLFlow
    """
    # Parse and log the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--l1_ratio", type=float, default=0.5)
    args = parser.parse_args()
    params = vars(args).items()
    for k, v in params:
        logging.info("%s: %s", k, v)
    # Load, train, evaluate and log the model
    x_train, x_test, y_train, y_test = load_data()
    model = train_model(x_train, y_train, args.alpha, args.l1_ratio)
    evaluate_model(model, x_test, y_test)


def load_data():
    """
    Load the data from the csv file and split it into train and test
    """
    logging.info("Loading data")
    data_path = "data/red-wine-quality.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        logging.error("Error reading the dataset %s", e)
    # registering the data
    X = df.drop(["quality"], axis=1)
    y = df["quality"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    mlflow.log_artifact(data_path)
    return x_train, x_test, y_train, y_test


def train_model(x, y, alpha, l1_ratio):
    """
    Train the model with the given parameters
    """
    logging.info("Training model")
    logging.info("model %s", "linear")
    logging.info("normalize %s", "True")
    logging.info("alpha %s", alpha)
    logging.info("l1_ratio %s", l1_ratio)
    #with mlflow.start_run(experiment_id=experiment.experiment_id):
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(x, y)
    mlflow.sklearn.log_model(model, "EN_model")
    return model


def evaluate_model(model, x, y):
    """
    Evaluate the model with the given test data
    """
    logging.info("Evaluating model")
    metrics = {}
    y_pred = model.predict(x)
    metrics["rmse"] = mean_squared_error(y, y_pred)
    metrics["r2"] = r2_score(y, y_pred)
    metrics["mae"] = mean_absolute_error(y, y_pred)
    for k, v in metrics.items():
        logging.info("%s: %s", k, v)
        mlflow.log_metric(k, v)
    return metrics


if __name__ == "__main__":
    main()
