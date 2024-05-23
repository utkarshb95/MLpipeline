import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from mldesigner import command_component, Input, Output

# Define attributes for the component
@command_component(
    name="train_setpoint_prediction",
    version="1",
    display_name="Train Setpoint Prediction",
    description="train setpoint prediction for climatesense system",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
    code=os.path.join(Path(os.path.abspath(os.curdir))),
)
def train_component(
    input_folder: Input(),
    output_folder: Output(),
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
    from training import Trainer
    from sklearn.linear_model import LinearRegression

    # Instantiate the Trainer class
    train = Trainer(model=LinearRegression(), use_standardscaler=True, session_id='drive_no')

    # Load the input data
    input_file = get_file(input_folder)
    print(input_file)
    input_df = pd.read_csv(input_file)
    print("This is the training dataset: ", input_df)

    # MLflow logger
    with mlflow.start_run():
        mlflow.sklearn.autolog()    # Enable autologging

        # Training code
        trained_pipeline, mse, r2, signature = train.train(input_df, features, target, test_exclude=0, verbose=1)
        print(f"The MSE is {mse} and R2 score is {r2}")

        # Addition of custom metrics
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("R-squared", r2)
        mlflow.sklearn.log_model(trained_pipeline, "audi_trained_model", signature=signature)

    # Output of the component
    model_path = os.path.join(output_folder, "audi_trained_model")
    mlflow.sklearn.save_model(trained_pipeline, model_path)