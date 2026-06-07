import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(version_base=None, config_name="config", config_path=".")
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    env_manager = config["main"].get("env_manager", "local")

    with tempfile.TemporaryDirectory() as tmp_dir:

        # ---------------------------------------------------------
        # DOWNLOAD STEP
        # ---------------------------------------------------------
        if "download" in active_steps:
            _ = mlflow.run(
                "components/get_data",
                "main",
                env_manager=env_manager,
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        # ---------------------------------------------------------
        # BASIC CLEANING STEP
        # ---------------------------------------------------------
        if "basic_cleaning" in active_steps:
            logger.info("Running basic_cleaning step")

            _ = mlflow.run(
                "components/basic_cleaning",
                "main",
                env_manager=env_manager,
                parameters={
                    "input_artifact": config["etl"]["sample_artifact"],
                    "output_artifact": config["etl"]["clean_sample_artifact"],
                    "output_type": config["etl"]["clean_sample_type"],
                    "output_description": config["etl"]["clean_sample_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        # ---------------------------------------------------------
        # DATA CHECK STEP
        # ---------------------------------------------------------
        if "data_check" in active_steps:
            logger.info("Running data_check step")

            clean_path = os.path.abspath(
                os.path.join(os.getcwd(), "data", config["etl"]["clean_sample_artifact"])
            )

            _ = mlflow.run(
                "components/data_check",
                "main",
                env_manager=env_manager,
                parameters={
                    "csv": clean_path,
                    "ref": clean_path,
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        # ---------------------------------------------------------
        # TRAIN/VAL/TEST SPLIT STEP
        # ---------------------------------------------------------
        if "data_split" in active_steps:
            logger.info("Running data_split step")

            os.environ["WANDB_MODE"] = "offline"

            _ = mlflow.run(
                "components/train_val_test_split",
                "main",
                env_manager=env_manager,
                parameters={
                    "input": config["etl"]["clean_sample_artifact"] + ":latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"].get("stratify_by", "none"),
                },
            )

        # ---------------------------------------------------------
        # TRAIN RANDOM FOREST STEP
        # ---------------------------------------------------------
        if "train_random_forest" in active_steps:

            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            logger.info("Running train_random_forest step")

            _ = mlflow.run(
                "components/train_random_forest",
                "main",
                env_manager=env_manager,
                parameters={
                    # FIXED: correct filename
                    "trainval_artifact": "trainval.csv:latest",

                    "val_size": config["modeling"].get("val_size", 0.2),
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": "none",
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"].get("max_tfidf_features", 5),
                    "output_artifact": config["modeling"].get(
                        "output_artifact_name", "random_forest_export"
                    ),
                },
            )

if __name__ == "__main__":
    go()