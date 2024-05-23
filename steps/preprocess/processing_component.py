import os
import sys
from pathlib import Path
from mldesigner import command_component, Input, Output
import csv
import pandas as pd

# Define attributes for the component
@command_component(
    name="preprocessor_for_audi_drives",
    version="1",
    display_name="Preprocess for Setpoint Prediction",
    description="preprocess data for setpoint prediction for climatesense system",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
    code=os.path.join(Path(os.path.abspath(os.curdir))),
)
def preprocessing_node(
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
    input_features = input_columns.split(",")
    target = target_columns.split(",")
    print(f"These are the input: {input_features}\n and this is the target: {target}")

    # Dependencies for the component functionality
    from preprocessing import Preprocessor
    from utilities import get_file

    # Processing audi data, define arguments as per use case
    pivot_columns = {
    'index': ['timestamp', 'drive_no'],
    'columns': 'signal_name',
    'values': 'signal_value'
    }

    clip_columns = {
        'tSetGblUsr_R1L_IHAL_d_C_G': (15, 30)
    }

    fill_method = 'ffill'  # or 'bfill'

    # Load the ingested data
    input_file = get_file(input_folder)
    input_df = pd.read_csv(input_file)

    # Initialize preprocessor object with appropriate arguments and process the data
    pre = Preprocessor(input_features, target, pivot_columns, fill_method, clip_columns)
    df = pre.clean_df(input_df)
    print("***This is the processed dataset***  ",df)

    # Save the processed data, component output
    path = os.path.join(output_folder, "audi_cleaned_data.csv")
    df.to_csv(path)