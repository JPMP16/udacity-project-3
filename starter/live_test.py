# Test live API on Heroku
import requests
import json

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

response = requests.post(url="https://udacity-project-3-jp2.herokuapp.com/infer/", data=json.dumps(data_dict))

print(response.status_code)
print(response.json())
