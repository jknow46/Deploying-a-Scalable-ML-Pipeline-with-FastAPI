import argparse
import logging
import os
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):
    run = wandb.init(job_type="get_data")
    logger.info(f"Returning sample {args.sample}")

    # Just log the provided sample file as artifact
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(args.sample)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("sample", type=str)
    parser.add_argument("artifact_name", type=str)
    parser.add_argument("artifact_type", type=str)
    parser.add_argument("artifact_description", type=str)

    args = parser.parse_args()
    go(args)