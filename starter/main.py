# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference
import pickle

app = FastAPI()


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
                "marital_status": "Divorced",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 60,
                "native_country": "United-States"
            }
        }


@app.get("/")
async def welcome():
    return {"message": "Welcome to the API"}


@app.post("/infer")
async def model_inference(data: Input):
    # Read inference artifacts
    with open('model/rf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)
    with open('model/encoder.pkl', 'rb') as f:
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
    input_data = process_data(data, categorical_features=cat_features, lb=lb, encoder=encoder, training=False)
    pred = inference(model, input_data)
    return {"prediction": pred}
