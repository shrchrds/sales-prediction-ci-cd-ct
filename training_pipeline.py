# training_pipeline.py

import os
import json
from datetime import datetime
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Default configs (can be overridden with environment variables)
DEFAULT_DATA_PATH = os.getenv("DATA_PATH", "data/add.csv")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
DEFAULT_TEST_SIZE = float(os.getenv("TEST_SIZE", "0.4"))
DEFAULT_RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def load_data(data_path: str = DEFAULT_DATA_PATH):
    """
    Load dataset and split into features and target.
    Expects columns: TV, Radio, Newspaper, Sales
    """
    data = pd.read_csv(data_path)

    X = data[["TV", "Radio", "Newspaper"]]
    y = data["Sales"]

    return X, y


def train_and_evaluate(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Train Linear Regression model and save it to disk.
    Returns metrics as a dict so that scripts / workflows can reuse them.
    """
    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "r2": float(r2),
        "mse": float(mse),
        "data_path": data_path,
        "model_path": model_path,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    return metrics


if __name__ == "__main__":
    metrics = train_and_evaluate()
    print("Model trained and saved to:", DEFAULT_MODEL_PATH)
    print("Training metrics:")
    print(json.dumps(metrics, indent=2))
