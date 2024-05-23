import datetime
import os
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential

# Connect to the workspace
deploy_local = False
credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")
subscription_id = "c8bc983f-b63d-431d-8119-f99bbb35af78"
rg_name = "cvateam"
workspace_name = "cvateam_MLDev"

ml_client = MLClient(credential, subscription_id, rg_name, workspace_name)

# Registered model
registered_model_name = "Audi_tSetGbl"
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
)
print("Model version: ", latest_model_version)

# Retrieve registered model
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)

# Environment for inferencing
environment = Environment(
    conda_file="endpoint_env.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
)

# Below code is to be used only when creating an endpoint

# Generate unique endpoint name
# endpoint_name = "audi-tsetgbl-endpoint-" + datetime.datetime.now().strftime("%m%d%H%M")
# Create online endpoint
# endpoint = ManagedOnlineEndpoint(
#     name=endpoint_name,
#     description="An online endpoint to generate predictions for the Audi Etron dataset",
#     auth_mode="key",
#     tags={"tSet": "R1L"},
# )
# ml_client.begin_create_or_update(endpoint)

# Specify the endpoint that needs to be used for deployment
endpoint_name = "audi-tsetgbl-endpoint-08221250"

# Create deployment
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=environment,
    code_configuration=CodeConfiguration(code="src", scoring_script="score.py"),
    instance_type="Standard_DS1_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(blue_deployment)

# Allocate traffic
# endpoint.traffic = { blue_deployment_name: 100 }
# ml_client.begin_create_or_update(endpoint).result()