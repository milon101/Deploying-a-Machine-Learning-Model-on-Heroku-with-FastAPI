#!/usr/bin/env python
"""
Upload the raw dataset in weights & biases
"""
import argparse
import logging
import wandb
import os

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="upload")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        os.path.join("data", args.sample),
        run
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Raw dataset upload")


    parser.add_argument(
        "--sample", 
        type=str,
        help="Name of the sample to upload",
        required=True
    )

    parser.add_argument(
        "--artifact_name", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type", 
        type=str,
        help="Type of the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description", 
        type=str,
        help="Description of the artifact",
        required=True
    )


    args = parser.parse_args()

    go(args)
