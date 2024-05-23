# Audi Etron ClimateSense Setpoint Prediction Model

Welcome to the repository containing the setpoint prediction model for the Audi Etron ClimateSense system.

## Folder Structure

The repository is organized as follows:

### src

This folder holds crucial source files:

- `utilities.py`: Responsible for loading and retrieving files.
- `preprocessing.py`: Performs data preprocessing on the Audi dataset for model training.
- `training.py`: Contains the code for training the prediction model.
- `model_testing.py`: Used to evaluate the trained model's performance on unseen data.

### steps

This directory utilizes the source files from the `src` folder to generate Azure ML components via the Azure SDK.

### notebooks

Explore the `notebooks` folder to find notebooks detailing analysis and experimental work.

### inference

The `inference` folder includes:

- Inference code
- Endpoint management
- Model deployment scripts

## Pipeline

The main pipeline initiation is governed by `pipeline.py`. To begin a job:

1. Make sure you're logged in by running the command:
   
    ```bash
    az login
    ```

2. Ensure the appropriate compute resource is ready and specified in pipeline.py to use the correct resource.

3. Trigger the pipeline with the command:

    ```bash
    python pipeline.py
    ```