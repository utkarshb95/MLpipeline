import os
import sys
from pathlib import Path
import pandas as pd
import joblib
from mldesigner import command_component, Input, Output

# Define attributes for the component
@command_component(
    name="test_setpoint_prediction",
    version="1",
    display_name="Test Setpoint Prediction",
    description="test setpoint prediction for climatesense system",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
    code=os.path.join(Path(os.path.abspath(os.curdir))),
)
def test_node(
    input_pkl_folder: Input(),
    output_model: Output(),
    input_folder: Input(),
    input_columns: str,
    target_columns: str,
):
    # Define the root folder path for the src folder
    root_folder_path = os.path.join(
        Path(os.path.abspath(os.curdir)).parent, "wd", "src"
    )
    sys.path.insert(1, root_folder_path)

    # Get list of features and target column
    features = input_columns.split(",")
    target = target_columns.split(",")
    print(f"These are the input: {features}\n and this is the target: {target}")

    # Dependencies for the component functionality
    import mlflow
    from utilities import get_file
    from model_testing import Tester

    # Load the input data
    input_file = get_file(input_folder)
    print(input_file)
    input_df = pd.read_csv(input_file)
    print(input_df)

    # Load the trained model
    model_path = os.path.join(input_pkl_folder, "audi_trained_model")
    loaded_model = mlflow.sklearn.load_model(model_path)
    print(loaded_model)

    # Model testing code
    test = Tester(loaded_model, session_id='drive_no') 
    acc = test.test(input_df, features, target, test_exclude=0, verbose=1)  # specify the session (test_exclude) to be tested on 
    print(acc)
    mlflow.log_metric("accuracy", acc)  # Log the test result