"""
    Description: This is a simple example of a machine learning model 
    using the iris dataset and MLFlow
"""
import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    """
    main function to train a model and log the results with MLFlow
    """
    mlflow.set_experiment("iris")
    mlflow.autolog()
    # load iris dataset
    x, y = datasets.load_iris(return_X_y=True)
    # split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    # train the model
    model = LogisticRegression(
        random_state=42, max_iter=1000, multi_class="multinomial"
    )
    model.fit(x_train, y_train)
    # evaluate the model
    accuracy = model.score(x_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
