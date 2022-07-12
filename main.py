from ensurepip import version
import json
from random import sample

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "upload",
    "basic_cleaning",
    # "data_check",
    "data_split",
    "train_gradient_boosting",
    "slice_performance_check",
    "model_check",
    # "api_creation"
]

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "upload" in active_steps:
            # upload the file in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "upload"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "census.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # download, clean and upload the file in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "census.csv:latest",
                    "output_artifact": "clean_census.csv",
                    "output_type": "clean_census",
                    "output_description": "Data with spaces removed from column names"
                }
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_census.csv:latest",
                }
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_val_test_split"),
                "main",
                parameters={
                    "input": "clean_census.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_gradient_boosting" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            gbc_config = os.path.abspath("gbc_config.json")
            with open(gbc_config, "w+") as fp:
                json.dump(dict(config["modeling"]["gradient_boosting_classifier"].items()), fp)  # DO NOT TOUCH

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_gradient_boosting"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "gbc_config": gbc_config,
                    "output_artifact_gbc": "gradient_boosting_export",
                    "output_artifact_lb": "label_binarizer_export"

                }
            )

        if "slice_performance_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "slice_performance_check"),
                "main",
                parameters={
                    "mlflow_model_gbc": "gradient_boosting_export:latest",
                    "mlflow_model_lb": "label_binarizer_export:latest",
                    "test_dataset": "test_data.csv:latest"
                }
            )

        if "model_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "model_check"),
                "main",
            )

        if "api_creation" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "api_creation"),
                "main",
                parameters={
                    "mlflow_model_gbc": "gradient_boosting_export:latest",
                    "mlflow_model_lb": "label_binarizer_export:latest",
                }
            )

if __name__ == "__main__":
    go()