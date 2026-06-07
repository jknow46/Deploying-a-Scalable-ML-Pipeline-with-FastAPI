#!/usr/bin/env python
import argparse
import logging
import os
import json
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def get_local_fallback_path(input_artifact: str):
    # strip :latest
    filename = input_artifact.split(":")[0]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "data", filename)


def load_trainval_csv(run, input_artifact: str):
    """
    Try W&B artifact, but always fall back to local file in offline mode.
    """
    try:
        logger.info(f"Trying to fetch artifact {input_artifact} from W&B")
        artifact = run.use_artifact(input_artifact)
        return artifact.file()
    except Exception:
        logger.warning("W&B offline — using local fallback file.")
        local_path = get_local_fallback_path(input_artifact)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local fallback file not found: {local_path}")
        return local_path


def go(args):
    run = wandb.init(
        project="nyc_airbnb",
        job_type="train_random_forest",
        mode=os.environ.get("WANDB_MODE", "online")
    )
    run.config.update(vars(args))

    # Load CSV (local fallback)
    csv_path = load_trainval_csv(run, args.trainval_artifact)
    df = pd.read_csv(csv_path)

    # Separate target
    y = df["price"]
    X = df.drop(columns=["price"])

    # Keep only numeric columns (simple fix for text columns)
    X = X.select_dtypes(include=["number"])

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.random_seed
    )

    # Load RF params
    with open(args.rf_config, "r") as fp:
        rf_params = json.load(fp)

    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "model.joblib")
    logger.info("Saved model.joblib")

    # Log artifact (works offline)
    artifact = wandb.Artifact(args.output_artifact, type="model_export")
    artifact.add_file("model.joblib")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_artifact", required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--stratify_by", default="none")
    parser.add_argument("--rf_config", required=True)
    parser.add_argument("--max_tfidf_features", type=int, default=5)
    parser.add_argument("--output_artifact", required=True)
    args = parser.parse_args()
    go(args)
