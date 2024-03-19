"""
Main script to train a model and log the results with MLFlow
"""
import argparse
import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
EXPERIMENT_NAME = "wine-quality2"
MLFLOW_TRACKING_SERVER_URI = os.environ.get("LOCAL_TRACKING_SERVER_URL")

# This uri can be a path or a server
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
    # azure_auth()
    logging.info("Using the MLFlow tracking server: %s",
                 mlflow.get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()
    # Load, train, evaluate and log the model
    with mlflow.start_run():
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
    x = df.drop(["quality"], axis=1)
    y = df["quality"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
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
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(x, y)
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
    logging.info("MLFlow run: %s", mlflow.active_run().info.run_id)
    return metrics


def azure_auth():
    """
    Authenticate to Azure
    """
    logging.info("Authenticating to Azure")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:  # pylint: disable=broad-except
        logging.error("Could not authenticate to Azure: %s", ex)
        credential = InteractiveBrowserCredential()
    return credential


if __name__ == "__main__":
    main()
