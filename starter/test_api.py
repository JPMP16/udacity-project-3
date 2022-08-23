import json
from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)


# Test get status code and response message
def test_get():
    r = client.get("/")
    assert r.json()["message"] == "Welcome to the API"
    assert r.status_code == 200


# Test post that has a lower than 50k salary
def test_post_lower():
    data_dict = {"age": 20,
                 "workclass": "Private",
                 "fnlgt": 215646,
                 "education": "HS-grad",
                 "education_num": 9,
                 "marital-status": "Divorced",
                 "occupation": "Exec-managerial",
                 "relationship": "Husband",
                 "race": "White",
                 "sex": "Male",
                 "capital-gain": 2174,
                 "capital-loss": 0,
                 "hours-per-week": 60,
                 "native-country": "United-States"
                 }
    data = json.dumps(data_dict)
    r = client.post("/infer", data=data)
    assert r.json()["prediction"] == "Salary <=50K"
    assert r.status_code == 200


# Test post that has a higher than 50k salary
def test_post_higher():
    data_dict = {"age": 40,
                 "workclass": "Private",
                 "fnlgt": 215646,
                 "education": "HS-grad",
                 "education_num": 9,
                 "marital-status": "Married",
                 "occupation": "Exec-managerial",
                 "relationship": "Husband",
                 "race": "White",
                 "sex": "Male",
                 "capital-gain": 21740,
                 "capital-loss": 1000,
                 "hours-per-week": 40,
                 "native-country": "United-States"
                 }
    data = json.dumps(data_dict)
    r = client.post("/infer", data=data)
    assert r.json()["prediction"] == "Salary >50K"
    assert r.status_code == 200
