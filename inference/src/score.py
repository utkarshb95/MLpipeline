import logging
import os
import json
import mlflow
from io import StringIO
from mlflow.pyfunc.scoring_server import infer_and_parse_json_input, predictions_to_json

def init():
    global model
    global input_schema
    # "model" is the path of the mlflow artifacts when the model was registered. For automl
    # models, this is generally "mlflow-model".
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    print(model_path)
    model = mlflow.pyfunc.load_model(model_path)
    input_schema = model.metadata.get_input_schema()
    logging.info('init completed.')

def run(raw_data):
    # Loads the data, runs inference and return the prediction.
    json_data = json.loads(raw_data)
    data = infer_and_parse_json_input(json_data, input_schema)
    predictions = model.predict(data)
    result = StringIO()
    predictions_to_json(predictions, result)
    return result.getvalue()