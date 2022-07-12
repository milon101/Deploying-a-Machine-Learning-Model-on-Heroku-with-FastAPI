#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import fbeta_score, precision_score, recall_score

# from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    gbc_model_local_path = run.use_artifact(args.mlflow_model_gbc).download()
    lb_model_local_path = run.use_artifact(args.mlflow_model_lb).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    # y_test = X_test.pop("salary")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(gbc_model_local_path)
    lb = mlflow.sklearn.load_model(lb_model_local_path)

    # y_test = lb.transform(y_test)
    y_pred = sk_pipe.predict(X_test)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    space_length = 28
    line = "{} {} {} {} {}".format(
                "Category" + ' ' * (space_length - len("Category")),
                " Slice Name" + ' ' * (space_length - len(" Slice Name")),
                "Precision" + ' ' * (space_length - len("Precision")),
                "Recall" + ' ' * (space_length - len("Recall")),
                "Fbeta" + ' ' * (space_length - len("Fbeta"))
            )
    logging.info(line)
    slice_preds = [line]
    for cat in cat_features: 
        for cls in X_test[cat].unique():
            df_slice = X_test[X_test[cat] == cls]
            y_test = df_slice.pop("salary")
            y_test = lb.transform(y_test)
            y_pred_slice = sk_pipe.predict(df_slice)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred_slice)

            line = "{} {} {} {} {}".format(
                cat + ' ' * (space_length - len(cat)),
                cls + ' ' * (space_length - len(cls)),
                str(precision) + ' ' * (space_length - len(str(precision))),
                str(recall) + ' ' * (space_length - len(str(recall))),
                str(fbeta) + ' ' * (space_length - len(str(fbeta)))
            )
            logging.info(line)
            slice_preds.append(line)

    with open('slice_output.txt', 'w') as out:
        for slice_pred in slice_preds:
            out.write(slice_pred + '\n')



    # logger.info("Scoring")
    # r_squared = sk_pipe.score(X_test, y_test)

    # mae = mean_absolute_error(y_test, y_pred)
    # precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    # logger.info(f"Precision: {precision}")
    # logger.info(f"Recall: {recall}")
    # logger.info(f"fbeta: {fbeta}")
    # logger.info(f"Score: {r_squared}")
    # logger.info(f"MAE: {mae}")

    # # Log MAE and r2
    # run.summary['precision'] = precision
    # run.summary['recall'] = recall
    # run.summary['fbeta'] = fbeta
    # # Here we save r_squared under the "r2" key
    # run.summary['r2'] = r_squared
    # # Now log the variable "mae" under the key "mae".
    # run.summary["mae"] = mae


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model_gbc",
        type=str, 
        help="Input Gradient Boosting MLFlow model",
        required=True
    )

    parser.add_argument(
        "--mlflow_model_lb",
        type=str, 
        help="Input Label Binarizer MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()
    logger.info(args)

    go(args)
