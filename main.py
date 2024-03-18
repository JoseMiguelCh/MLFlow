"""
Main script to train a model and log the results with MLFlow
"""
import argparse
import logging
import os
import mlflow
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)
TRACKING_URI = os.environ.get("LOCAL_TRACKING_SERVER_URL")
EXPERIMENT_NAME = "wine-quality"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

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
    mlflow.log_artifact(data_path)
    X = df.drop(["quality"], axis=1)
    y = df["quality"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def train_model(x, y, alpha, l1_ratio):
    """
    Train the model with the given parameters
    """
    logging.info("Training model")
    with mlflow.start_run(nested=True):
        model = None
        mlflow.log_param("model", "linear")
        mlflow.log_param("normalize", "True")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(x, y)
        datetime = mlflow.active_run().info.start_time
        modelpath = f"models/wine-quality-{datetime}"
        mlflow.sklearn.save_model(model, modelpath)
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
    mlflow.log_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()
