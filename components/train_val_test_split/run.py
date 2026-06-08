import argparse
import logging
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):
    run = wandb.init(job_type="train_val_test_split")

    logger.info("Downloading input artifact")
    artifact = run.use_artifact(args.input)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    stratify = df[args.stratify_by] if args.stratify_by != "none" else None

    trainval_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify,
    )

    trainval_df.to_csv("trainval.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    trainval_art = wandb.Artifact(
        "trainval.csv",
        type="trainval_data",
        description="Train/validation split",
    )
    trainval_art.add_file("trainval.csv")
    run.log_artifact(trainval_art)

    test_art = wandb.Artifact(
        "test.csv",
        type="test_data",
        description="Test split",
    )
    test_art.add_file("test.csv")
    run.log_artifact(test_art)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)

    args = parser.parse_args()
    go(args)