#!/usr/bin/env python
"""
Train/val/test split component runner.

This version is robust to W&B offline mode: it will try to use a W&B artifact,
and if that fails (offline mode or artifact not available) it will fall back
to a local file under data/<artifact_name>.
"""
import argparse
import logging
import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def get_local_fallback_path(input_artifact: str):
    filename = input_artifact.split(":")[0]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_path = os.path.join(project_root, "data", filename)
    return local_path


def load_input_csv(run, input_artifact: str):
    try:
        logger.info("Attempting to fetch artifact %s from W&B", input_artifact)
        artifact = run.use_artifact(input_artifact)
        path = artifact.file()
        logger.info("Fetched artifact from W&B: %s", path)
        return path
    except TypeError:
        logger.warning("W&B offline mode detected or use_artifact not allowed. Falling back to local file.")
    except Exception as e:
        logger.warning("Could not fetch artifact from W&B (%s). Falling back to local file.", e)

    local_path = get_local_fallback_path(input_artifact)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local fallback file not found: {local_path}")
    logger.info("Using local fallback file: %s", local_path)
    return local_path


def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="train_val_test_split", mode=os.environ.get("WANDB_MODE", "online"))
    run.config.update(vars(args))

    csv_path = load_input_csv(run, args.input)
    df = pd.read_csv(csv_path)

    stratify_col = None
    if args.stratify_by and args.stratify_by.lower() != "none":
        if args.stratify_by not in df.columns:
            logger.warning("Stratify column %s not in dataframe; proceeding without stratify.", args.stratify_by)
        else:
            stratify_col = df[args.stratify_by]

    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_col,
    )

    trainval_path = "trainval.csv"
    test_path = "test.csv"
    trainval.to_csv(trainval_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info("Saved train/val to %s and test to %s", trainval_path, test_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_trainval = os.path.join(project_root, "data", "trainval.csv")
    out_test = os.path.join(project_root, "data", "test.csv")
    os.makedirs(os.path.dirname(out_trainval), exist_ok=True)
    trainval.to_csv(out_trainval, index=False)
    test.to_csv(out_test, index=False)
    logger.info("Saved copies to %s and %s", out_trainval, out_test)

    artifact_trainval = wandb.Artifact("trainval", type="dataset")
    artifact_trainval.add_file(trainval_path)
    run.log_artifact(artifact_trainval)

    artifact_test = wandb.Artifact("test", type="dataset")
    artifact_test.add_file(test_path)
    run.log_artifact(artifact_test)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input artifact (e.g. clean_sample.csv:latest)")
    parser.add_argument("test_size", type=float, help="Test size fraction (e.g. 0.2)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stratify_by", type=str, default="none", help="Column to stratify by or 'none'")
    args = parser.parse_args()
    go(args)