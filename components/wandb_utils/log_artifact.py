import wandb
import mlflow


def log_artifact(artifact_name, artifact_type, artifact_description, filename, wandb_run):
    """
    Log the provided filename as an artifact in W&B, and add the artifact path to the MLFlow run
    so it can be retrieved by subsequent steps in a pipeline.

    :param artifact_name: name for the artifact
    :param artifact_type: type for the artifact (e.g. "raw_data", "clean_data")
    :param artifact_description: a brief description of the artifact
    :param filename: local filename for the artifact
    :param wandb_run: current Weights & Biases run
    :return: None
    """

    # Create the W&B artifact
    artifact = wandb.Artifact(
        artifact_name,
        type=artifact_type,
        description=artifact_description,
    )

    # Add the file to the artifact
    artifact.add_file(filename)

    # Log the artifact to the current run
    wandb_run.log_artifact(artifact)
