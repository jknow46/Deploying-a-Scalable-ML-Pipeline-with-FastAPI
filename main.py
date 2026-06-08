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
    "test_regression_model",
]


@hydra.main(version_base=None, config_name="config", config_path=".")
def go(config: DictConfig):

    # Always work from original project root
    os.chdir(hydra.utils.get_original_cwd())

    os.environ["WANDB_PROJECT"] = config.main.project_name
    os.environ["WANDB_RUN_GROUP"] = config.main.experiment_name

    steps_param = config.main.steps
    active_steps = steps_param.split(",") if steps_param != "all" else _steps

    env_manager = config.main.get("env_manager", "local")

    with tempfile.TemporaryDirectory() as tmp_dir:

        # DOWNLOAD
        if "download" in active_steps:
            mlflow.run(
                "components/get_data",
                "main",
                env_manager=env_manager,
                parameters={
                    "sample": config.etl.sample,
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        # BASIC CLEANING
        if "basic_cleaning" in active_steps:
            mlflow.run(
                "components/basic_cleaning",
                "main",
                env_manager=env_manager,
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": config.etl.clean_sample_artifact,
                    "output_type": config.etl.clean_sample_type,
                    "output_description": config.etl.clean_sample_description,
                    "min_price": config.etl.min_price,
                    "max_price": config.etl.max_price,
                },
            )

        # DATA CHECK
        if "data_check" in active_steps:
            clean_path = os.path.abspath(
                os.path.join(os.getcwd(), "data", config.etl.clean_sample_artifact)
            )

            mlflow.run(
                "components/data_check",
                "main",
                env_manager=env_manager,
                parameters={
                    "csv": clean_path,
                    "ref": clean_path,
                    "kl_threshold": config.data_check.kl_threshold,
                    "min_price": config.etl.min_price,
                    "max_price": config.etl.max_price,
                },
            )

        # TRAIN/VAL/TEST SPLIT
        if "data_split" in active_steps:
            mlflow.run(
                "components/train_val_test_split",
                "main",
                env_manager=env_manager,
                parameters={
                    "input": config.etl.clean_sample_artifact + ":latest",
                    "test_size": config.modeling.test_size,
                    "random_seed": config.modeling.random_seed,
                    "stratify_by": config.modeling.get("stratify_by", "none"),
                },
            )

        # TRAIN RANDOM FOREST
        if "train_random_forest" in active_steps:

            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w") as fp:
                json.dump(dict(config.modeling.random_forest.items()), fp)

            mlflow.run(
                "components/train_random_forest",
                "main",
                env_manager=env_manager,
                parameters={
                    "trainval_artifact": "trainval.csv:latest",
                    "rf_config": rf_config,
                    "output_artifact": config.modeling.get(
                        "output_artifact_name", "random_forest_export"
                    ),
                },
            )

        # TEST REGRESSION MODEL
        if "test_regression_model" in active_steps:
            mlflow.run(
                "components/test_regression_model",
                "main",
                env_manager=env_manager,
                parameters={
                    "model_export": config.modeling.get(
                        "output_artifact_name", "random_forest_export"
                    ) + ":latest",
                    "test_data": "test.csv:latest",
                },
            )


if __name__ == "__main__":
    go()