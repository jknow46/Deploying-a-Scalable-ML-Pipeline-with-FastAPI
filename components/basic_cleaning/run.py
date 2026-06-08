import argparse
import logging
import wandb
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # ---------------------------------------------------------
    # NYC bounding box filter (required for v1.0.1)
    # ---------------------------------------------------------
    min_longitude, max_longitude = -74.25, -73.50
    min_latitude, max_latitude = 40.5, 41.2

    df = df[
        df["longitude"].between(min_longitude, max_longitude)
        & df["latitude"].between(min_latitude, max_latitude)
    ]

    # ---------------------------------------------------------
    # Price filter
    # ---------------------------------------------------------
    logger.info("Filtering price outliers")
    df = df[df["price"].between(args.min_price, args.max_price)]

    # ---------------------------------------------------------
    # Save cleaned data
    # ---------------------------------------------------------
    cleaned_path = Path("clean_sample.csv")
    df.to_csv(cleaned_path, index=False)

    logger.info("Uploading cleaned artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(cleaned_path)
    run.log_artifact(artifact)

    logger.info("Basic cleaning completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning of Airbnb data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input raw data artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the cleaned artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the cleaned artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price allowed",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price allowed",
        required=True,
    )

    args = parser.parse_args()

    go(args)