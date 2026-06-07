#!/usr/bin/env python
"""
Wrapper to run the data_check pytest suite programmatically so conftest.py
is discovered and custom pytest options (--csv, --ref, etc.) are registered.
"""

import argparse
import os
import sys
import pytest
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run data_check pytest suite")
    parser.add_argument("--csv", required=True, help="Path to CSV to test")
    parser.add_argument("--ref", required=True, help="Path to reference CSV")
    parser.add_argument("--kl_threshold", type=float, required=True, help="KL threshold")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum price")
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure the test folder path is absolute
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tests_path = os.path.join(project_root, "components", "data_check")

    # Build pytest args: run tests in components/data_check and pass the custom options
    pytest_args = [
        "-vv",
        tests_path,
        "--csv", args.csv,
        "--ref", args.ref,
        "--kl_threshold", str(args.kl_threshold),
        "--min_price", str(args.min_price),
        "--max_price", str(args.max_price),
    ]

    logger.info("Running pytest with args: %s", " ".join(pytest_args))

    # Call pytest programmatically. This will load conftest.py from tests_path.
    exit_code = pytest.main(pytest_args)

    # Propagate pytest exit code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()