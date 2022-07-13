

from fastapi.testclient import TestClient
from api import app
import pytest

@pytest.fixture
def client():
    """
    Get test client
    """
    client = TestClient(app)
    return client


def test_get_path(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Deploying a Machine Learning Model on Heroku with FastAPI"}


def test_inference_less(client):
    r = client.post("/", json={
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
    })
    assert r.status_code == 200
    print(r.json())
    assert r.json() == {"prediction": "<=50K"}

def test_inference_greater(client):
    r = client.post("/", json={
        "age": 31,
        "workclass": "Private",
        "fnlwgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    })
    assert r.status_code == 200
    print(r.json())
    assert r.json() == {"prediction": ">50K"}

# if __name__ == "__main__":
#     inference()
