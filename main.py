# API code
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference
import pickle
import json
import pandas as pd

app = FastAPI()


# Input data model
class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 20,
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
        }

# Welcome message
@app.get("/")
async def welcome():
    return {"message": "Welcome to the API"}


# Predict Salary
@app.post("/infer")
async def model_inference(data: Input):
    # Read data
    cols = ["age",
            "workclass",
            "fnlgt",
            "education",
            "education_num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country"]
    data_dict = json.loads(data.json())
    df = pd.DataFrame({k: [v] for k, v in data_dict.items()})
    df.columns = cols
    # Load inference artifacts
    with open('starter/model/rf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('starter/model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)
    with open('starter/model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Inference Pipeline
    input_data, _, _, _ = process_data(df, categorical_features=cat_features, lb=lb, encoder=encoder, training=False)
    pred = inference(model, input_data)
    # Result string
    if pred[0] == 0:
        result = 'Salary <=50K'
    else:
        result = 'Salary >50K'
    return {"prediction": result}
