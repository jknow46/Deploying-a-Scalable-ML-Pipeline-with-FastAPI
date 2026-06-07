#!/usr/bin/env python
"""
This script returns a local sample file and logs it as a W&B artifact.
"""
import argparse
import logging
import os

import wandb
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="download_file", mode="online")
    run.config.update(args)

    logger.info(f"Returning sample {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")

    # FIX: compute absolute path to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_file = os.path.join(project_root, "data", args.sample)

    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        local_file,
        run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Return a local sample file")

    parser.add_argument("sample", type=str, help="Name of the sample file")
    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("artifact_type", type=str, help="Output artifact type")
    parser.add_argument("artifact_description", type=str, help="Artifact description")

    args = parser.parse_args()

    go(args)
