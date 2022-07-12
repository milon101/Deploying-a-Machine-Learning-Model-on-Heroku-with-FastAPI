#!/usr/bin/env python
"""
Remove spaces from the column name
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np

# from src.wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def log_artifact(artifact_name, artifact_type, artifact_description, filename, wandb_run):
    """
    Log the provided filename as an artifact in W&B, and add the artifact path to the MLFlow run
    so it can be retrieved by subsequent steps in a pipeline

    :param artifact_name: name for the artifact
    :param artifact_type: type for the artifact (just a string like "raw_data", "clean_data" and so on)
    :param artifact_description: a brief description of the artifact
    :param filename: local filename for the artifact
    :param wandb_run: current Weights & Biases run
    :return: None
    """
    # Log to W&B
    artifact = wandb.Artifact(
        artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Downloading {args.input_artifact} from Weights & Biases")
    artifact_loacl_path = run.use_artifact(args.input_artifact).file()
    data = pd.read_csv(artifact_loacl_path, skipinitialspace=True)

    data.columns = data.columns.str.replace(' ','')

    data= data.replace('\?', np.nan, regex=True)
    data.dropna(inplace=True)
    
    data.to_csv(args.output_artifact, index=False)

    logger.info(f"Uploading {args.output_artifact} to Weights & Biases")
    log_artifact(
        args.output_artifact,
        args.output_type,
        args.output_description,
        "clean_census.csv",
        run
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning raw data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact to download",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )


    args = parser.parse_args()

    go(args)
