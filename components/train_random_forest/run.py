import argparse
import logging
import json
import wandb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):

    run = wandb.init(job_type="train_random_forest")

    logger.info("Downloading trainval artifact")
    artifact = run.use_artifact(args.trainval_artifact)
    artifact_path = artifact.file()
    logger.info(f"Artifact downloaded to: {artifact_path}")

    df = pd.read_csv(artifact_path)

    # Load RF config
    with open(args.rf_config, "r") as fp:
        rf_config = json.load(fp)

    # Split target
    y = df["price"]
    X = df.drop(columns=["price"])

    # ⭐ NEW: Use ONLY numeric features (consistent between train & test)
    logger.info("Selecting numeric features only")
    X = X.select_dtypes(include=["number"])

    # Train model
    logger.info("Training Random Forest")
    rf = RandomForestRegressor(**rf_config)
    rf.fit(X, y)

    # Save model
    joblib.dump(rf, "model.joblib")

    # Log model
    model_artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random forest model",
    )
    model_artifact.add_file("model.joblib")
    run.log_artifact(model_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)

    args = parser.parse_args()
    go(args)