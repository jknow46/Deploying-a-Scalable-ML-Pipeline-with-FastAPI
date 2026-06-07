#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        project="nyc_airbnb",
        job_type="basic_cleaning",
        group="cleaning",
        save_code=True,
        mode="online"
    )
    run.config.update(args)

    # Try normal artifact loading; fallback to local file in offline mode
    try:
        artifact = run.use_artifact(args.input_artifact)
        artifact_local_path = artifact.file()
    except Exception:
        # compute absolute path to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        filename = args.input_artifact.split(":")[0]  # sample.csv

        # Map artifact name → real file
        if filename == "sample.csv":
            filename = "sample1.csv"

        artifact_local_path = os.path.join(project_root, "data", filename)

    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save cleaned data to the component working directory
    df.to_csv("clean_sample.csv", index=False)
    logger.info("Saved cleaned data to component working directory as clean_sample.csv")

    # Also save a copy to the project data folder so other components can access it
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    shared_path = os.path.join(project_root, "data", args.output_artifact)
    os.makedirs(os.path.dirname(shared_path), exist_ok=True)
    df.to_csv(shared_path, index=False)
    logger.info(f"Saved cleaned data to {shared_path}")

    # Log cleaned artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info("Logged cleaned artifact to W&B")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact to download",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price allowed",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price allowed",
        required=True
    )

    args = parser.parse_args()

    go(args)