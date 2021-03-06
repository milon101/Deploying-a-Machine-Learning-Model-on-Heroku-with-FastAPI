import pytest
import pandas as pd
import logging
import os
import mlflow
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@pytest.fixture(scope='session')
def model():

    gbc_model_local_path = os.path.join("src/train_gradient_boosting/models", "gradient_boosting_dir")
    lb_model_local_path =  os.path.join("src/train_gradient_boosting/models", "label_binarizer_dir")

    return gbc_model_local_path, lb_model_local_path

def test_export_model(model):
    """
    Checks if the model exported
    """
    gbc_model_local_path, lb_model_local_path = model
    assert gbc_model_local_path
    assert lb_model_local_path

    try:
        sk_pipe = mlflow.sklearn.load_model(gbc_model_local_path)
        lb = mlflow.sklearn.load_model(lb_model_local_path)
    except mlflow.exceptions.MlflowException as err:
        logger.info("Could not find an sk_pipe configuration file at model path")
        raise err

def test_label_binarizer(model):
    """
    Test label binarizer
    """
    _, lb_model_local_path = model
    lb = mlflow.sklearn.load_model(lb_model_local_path)
    values = ["<=50K", ">50K"]
    v = lb.transform(values)
    assert (v == [[0], [1]]).all()

def test_inference_less(model):
    """
    Test inference 
    """

    gbc_model_local_path, lb_model_local_path = model
    logger.info("Loading model and performing inference")
    logger.info(gbc_model_local_path)
    sk_pipe = mlflow.sklearn.load_model(gbc_model_local_path)
    lb = mlflow.sklearn.load_model(lb_model_local_path)

    array = np.array([[
        39,
        "State-gov",
        77516,
        "Bachelors",
        13,
        "Never-married",
        "Adm-clerical",
        "Not-in-family",
        "White",
        "Male",
        0,
        2174,
        40,
        "United-States"]])

    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ])

    pred = sk_pipe.predict(df_temp)
    y = lb.inverse_transform(pred)[0]

    assert y == "<=50K"


def test_inference_greater(model):
    """
    Test inference 
    """

    gbc_model_local_path, lb_model_local_path = model
    logger.info("Loading model and performing inference")
    logger.info(gbc_model_local_path)
    sk_pipe = mlflow.sklearn.load_model(gbc_model_local_path)
    lb = mlflow.sklearn.load_model(lb_model_local_path)

    array = np.array([[
        31,
        "Private",
        45781,
        "Masters",
        14,
        "Never-married",
        "Prof-specialty",
        "Not-in-family",
        "White",
        "Female",
        14084,
        0,
        50,
        "United-States"
        ]])

    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ])

    pred = sk_pipe.predict(df_temp)
    y = lb.inverse_transform(pred)[0]

    assert y == ">50K"