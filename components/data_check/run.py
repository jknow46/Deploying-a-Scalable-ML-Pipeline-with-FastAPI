#!/usr/bin/env python

import argparse
import logging
import pandas as pd
import wandb
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):

    run = wandb.init(
        project="nyc_airbnb",
        job_type="data_check",
        save_code=True,
        mode="online"
    )
    run.config.update(vars(args))

    df = pd.read_csv(args.csv)
    ref = pd.read_csv(args.ref)

    pytest.main([
        "-q",
        "--disable-warnings",
        f"--csv={args.csv}",
        f"--ref={args.ref}",
        f"--kl_threshold={args.kl_threshold}",
        f"--min_price={args.min_price}",
        f"--max_price={args.max_price}",
        "tests/test_data.py"
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data check")

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--kl_threshold", type=float, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()
    go(args)