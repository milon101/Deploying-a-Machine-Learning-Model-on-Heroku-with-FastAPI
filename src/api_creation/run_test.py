

from fastapi.testclient import TestClient
from run import app
client = TestClient(app)

def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings!"}


def inference():
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

if __name__ == "__main__":
    inference()