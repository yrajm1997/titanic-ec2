from typing import Any, List, Optional

from pydantic import BaseModel
from titanic_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "PassengerId": 79,
                        "Pclass": 2,
                        "Name": "Caldwell, Master. Alden Gates",
                        "Sex": "male",
                        "Age": 0.83,
                        "SibSp": 0,
                        "Parch": 2,
                        "Ticket": "248738",
                        "Cabin": 'A5',
                        "Embarked": "S",
                        "Fare": 29,
                    }
                ]
            }
        }
