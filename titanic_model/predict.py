import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from titanic_model import __version__ as _version
from titanic_model.config.core import config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import load_pipeline
from titanic_model.processing.data_manager import pre_pipeline_preparation
from titanic_model.processing.validation import validate_inputs

#################### MLflow CODE START to load 'production' model #############################
import mlflow 
import mlflow.pyfunc
mlflow.set_tracking_uri("http://192.168.43.86:5000")

# Create MLflow client
client = mlflow.tracking.MlflowClient()

# Load model via 'models'
model_name = "sklearn-titanic-rf-model"
model_info = client.get_model_version_by_alias(model_name, "production")
print(f'Model version fetched: {model_info.version}')

titanic_pipe = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@production")
#################### MLflow CODE END ##########################################################


#pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
#titanic_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = titanic_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = titanic_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'PassengerId':[79],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
                'SibSp':[0],'Parch':[2],'Ticket':['248738'],'Cabin':[np.nan,],'Embarked':['S'],'Fare':[29]}
    
    make_prediction(input_data=data_in)
