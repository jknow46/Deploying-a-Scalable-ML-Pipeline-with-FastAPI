import argparse
import logging
import wandb
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):
    run = wandb.init(job_type="test_regression_model")

    logger.info("Downloading model artifact")
    model_art = run.use_artifact(args.model_export)
    model_path = model_art.file()

    logger.info("Downloading test data artifact")
    test_art = run.use_artifact(args.test_data)
    test_path = test_art.file()

    model = joblib.load(model_path)
    df = pd.read_csv(test_path)

    y = df["price"]
    X = df.drop(columns=["price"])

    # Use ONLY numeric features (matches training)
    logger.info("Selecting numeric features only")
    X = X.select_dtypes(include=["number"])

    # Predict
    y_pred = model.predict(X)

    # Compute RMSE manually (compatible with older sklearn)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    r2 = r2_score(y, y_pred)

    logger.info(f"RMSE: {rmse}")
    logger.info(f"R2: {r2}")

    wandb.log({"rmse": rmse, "r2": r2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_export", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)

    args = parser.parse_args()
    go(args)
