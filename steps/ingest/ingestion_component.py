# import required libraries
import os
import sys
from os.path import abspath, dirname
from pathlib import Path
import pandas as pd
from mldesigner import Input, Output, command_component

# Define attributes for the component
@command_component(
    name="ingest_data",
    version="1",
    display_name="Ingest data",
    description="Ingest data from csv file.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
    code=os.path.join(Path(os.path.abspath(os.curdir))),
)
def ingest_from_csv_component(
    input_folder: Input,
    output_folder: Output,
):
    # Define the root folder path for the src folder
    root_folder_path = os.path.join(
        Path(os.path.abspath(os.curdir)).parent, "wd", "src"
    )
    sys.path.insert(1, root_folder_path)

    # Dependencies for the component functionality
    from utilities import data_load

    # Ingest data from azure data asset and store in pipeline directory
    df = data_load(input_folder, derive_drive_id=True)
    print(df)

    # Output of the component
    path = os.path.join(output_folder, "audi_data.csv")
    df.to_csv(path)
