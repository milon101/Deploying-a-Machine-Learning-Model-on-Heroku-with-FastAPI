#!/usr/bin/env python
"""
This script trains a Gradient Boosting Classifier
"""
import argparse
import imp
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelBinarizer, FunctionTransformer
from sklearn.metrics import fbeta_score, precision_score, recall_score

import wandb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="train_gradient_boosting")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.gbc_config) as fp:
        gbc_config = json.load(fp)
    run.config.update(gbc_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    gbc_config['random_state'] = args.random_seed

    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("salary")  # this removes the column "salary" from X and puts it into y

    lb = LabelBinarizer()

    # logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=args.val_size, 
        stratify=X[args.stratify_by] if args.stratify_by != 'none' else None,
        random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    # sk_pipe, processed_features = get_inference_pipeline(gbc_config)
    y_train = lb.fit_transform(y_train)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    ######################################
    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train

    # sk_pipe.fit(X_train, y_train)
    sk_pipe, processed_features = train_model(gbc_config, X_train, y_train)

    ######################################

    # Compute r2 and MAE
    logger.info("Scoring")
    y_val = lb.transform(y_val)
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = inference(sk_pipe, X_val)
    precision, recall, fbeta = compute_model_metrics(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"fbeta: {fbeta}")
    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")


    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    dirs = ["models", "models/gradient_boosting_dir", "models/label_binarizer_dir"]
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)

    mlflow.sklearn.save_model(
        sk_pipe,
        dirs[1],
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:2]
        )

    mlflow.sklearn.save_model(
        lb,
        dirs[2],
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:2]
        )

    artifact_gbc = wandb.Artifact(
        args.output_artifact_gbc,
        type="model_export",
        description="Export Gradient Boosting Classifer model",
        metadata=gbc_config
    )
    artifact_gbc.add_dir(dirs[1])
    run.log_artifact(artifact_gbc)
    artifact_gbc.wait()

    artifact_lb = wandb.Artifact(
        args.output_artifact_lb,
        type="model_export",
        description="Export Gradient Boosting Classifer model",
        metadata=gbc_config
    )
    artifact_lb.add_dir(dirs[2])
    run.log_artifact(artifact_lb)
    artifact_lb.wait()
    ######################################

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    ######################################
    run.summary['precision'] = precision
    run.summary['recall'] = recall
    run.summary['fbeta'] = fbeta
    # Here we save r_squared under the "r2" key
    run.summary['r2'] = r_squared
    # Now log the variable "mae" under the key "mae".
    run.summary["mae"] = mae
    ######################################

    # Upload to W&B the feture importance visualization
    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )

def train_model(gbc_config, X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    sk_pipe, processed_features = get_inference_pipeline(gbc_config)
    sk_pipe.fit(X_train, y_train)
    return sk_pipe, processed_features




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


def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    # idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(gbc_config):

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
    categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    zero_imputed = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_preproc, cat_features),
            ("impute_zero", zero_imputer, zero_imputed),
            # ("lb", lb, label)
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = cat_features + zero_imputed

    # Create random forest
    random_Forest = GradientBoostingClassifier(**gbc_config)

    sk_pipe = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("random_forest", random_Forest)
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--gbc_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--output_artifact_gbc",
        type=str,
        help="Name for the gradient boosting output serialized model",
        required=True,
    )

    parser.add_argument(
        "--output_artifact_lb",
        type=str,
        help="Name for the label binarizer output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)


