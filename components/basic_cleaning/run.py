import argparse
import logging
import os
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):
    run = wandb.init(job_type="basic_cleaning")

    logger.info("Downloading input artifact")
    artifact = run.use_artifact(args.input_artifact)
    input_path = artifact.file()

    df = pd.read_csv(input_path)

    # Price filter
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # ⭐ NEW REQUIRED CLEANING STEP — NYC bounding box ⭐
    logger.info("Removing listings outside NYC bounding box")

    min_longitude = -74.25
    max_longitude = -73.50
    min_latitude = 40.5
    max_latitude = 41.2

    df = df[
        df["longitude"].between(min_longitude, max_longitude)
        & df["latitude"].between(min_latitude, max_latitude)
    ].copy()

    # Drop outliers / NA
    df = df.dropna().reset_index(drop=True)

    cleaned_path = "clean_sample.csv"
    df.to_csv(cleaned_path, index=False)

    out_art = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    out_art.add_file(cleaned_path)
    run.log_artifact(out_art)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()
    go(args)