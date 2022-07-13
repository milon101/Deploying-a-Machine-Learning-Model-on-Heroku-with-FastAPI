import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

data = json={
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 2174,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

r = requests.post("https://deploying-a-ml-model-with-fast.herokuapp.com/", json=data)

# assert r.status_code == 200
logger.info(f"Response status code: {r.status_code}")
logger.info(f"Response body: {r.json()}")