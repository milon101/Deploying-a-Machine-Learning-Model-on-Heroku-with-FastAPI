#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import logging
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import os 
import numpy as np
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

class Value(BaseModel):
    age: int
    workclass: Literal['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
       'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']
    fnlwgt: int
    education: Literal['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
       'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
       '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
    education_num: int
    marital_status: Literal['Never-married', 'Married-civ-spouse', 'Divorced',
       'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
       'Widowed']
    occupation: Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
       'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
       'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
       'Tech-support', '?', 'Protective-serv', 'Armed-Forces',
       'Priv-house-serv']
    relationship: Literal['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
       'Other-relative']
    race: Literal['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
       'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
       'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
       'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
       'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
       'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
       'China', 'Japan', 'Yugoslavia', 'Peru',
       'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
       'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
       'Holand-Netherlands']

app = FastAPI()

@app.get("/")
async def get_items():
    return {"message": "Greetings!"}

@app.post("/")
async def inference(user_data: Value):

   # run = wandb.init(job_type="test_model")
   # run.config.update(app.args)

   logger.info("Downloading artifacts")
   # logger.info(os.getcwd())
   # Download input artifact. This will also log that this script is using this
   # particular version of the artifact
   gbc_model_local_path = os.path.join("src/train_gradient_boosting/models", "gradient_boosting_dir")
   lb_model_local_path = os.path.join("src/train_gradient_boosting/models", "label_binarizer_dir")

   # gbc_model_local_path = run.use_artifact(app.args.mlflow_model_gbc).download()
   # lb_model_local_path = run.use_artifact(app.args.mlflow_model_lb).download()

   logger.info("Loading model and performing inference on test set")
   logger.info(gbc_model_local_path)
   sk_pipe = mlflow.sklearn.load_model(gbc_model_local_path)
   lb = mlflow.sklearn.load_model(lb_model_local_path)

   array = np.array([[
                     user_data.age,
                     user_data.workclass,
                     user_data.fnlwgt,
                     user_data.education,
                     user_data.education_num,
                     user_data.marital_status,
                     user_data.occupation,
                     user_data.relationship,
                     user_data.race,
                     user_data.sex,
                     user_data.capital_gain,
                     user_data.capital_loss,
                     user_data.hours_per_week,
                     user_data.native_country
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
   return {"prediction": y}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload = True, log_level="info")
