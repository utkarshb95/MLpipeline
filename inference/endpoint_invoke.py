import uuid
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)

# Connect to the workspace
deploy_local = False
credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")
subscription_id = "c8bc983f-b63d-431d-8119-f99bbb35af78"
rg_name = "cvateam"
workspace_name = "cvateam_MLDev"

ml_client = MLClient(credential, subscription_id, rg_name, workspace_name)

# Specify the endpoint name that needs to be invoked
endpoint_name = "audi-tsetgbl-endpoint-08221250"

# Submit request to the endpoint
response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name="blue",
    request_file="sample_request.json",
    local=deploy_local,
)

# To delete the endpoint
# ml_client.online_endpoints.begin_delete(name=endpoint_name)   
# ml_client.begin_create_or_update(endpoint_name).result()

# Print the prediction results
print("Prediction Results:")
print(response)