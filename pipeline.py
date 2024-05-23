# Import azure sdk dependencies
from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# Import steps of the pipeline
from steps.ingest.ingestion_component import ingest_from_csv_component
from steps.preprocess.processing_component import preprocessing_node
from steps.train.training_component import train_component
from steps.test.test_component import test_node

# Enter details of Azure Machine Learning workspace
cpu_compute_target = "ml-utkarsh-lite"
credential = DefaultAzureCredential()
subscription_id = "c8bc983f-b63d-431d-8119-f99bbb35af78"
rg_name = "cvateam"
workspace_name = "cvateam_MLDev"
try:
    ml_client = MLClient.from_config(credential)
except Exception as ex:
    print(ex)
    subscription_id = subscription_id
    resource_group_name = rg_name
    workspace_name = workspace_name
    ml_client = MLClient(
        credential, subscription_id, resource_group_name, workspace_name
    )
    print(ml_client.compute.get(cpu_compute_target))

# Source dataset path for ingestion
input = Input(
    type="uri_folder",
    path='azureml:AudiEtron2023:1',
)

# Pipeline decorator with pipeline attributes
@pipeline(
    default_compute=cpu_compute_target,
    description="Audi etron setpoint prediction.",
    tags={"version": "1", "testrun": True},
)
def audi_setpoint_regressor(
    input_folder: Input,
    input_columns: str,
    target_columns: str,
):
    """E2E training for audi etron project."""
    ingest_data_node = ingest_from_csv_component(
        input_folder=input_folder
    )
    processing_node = preprocessing_node(
        input_folder = ingest_data_node.outputs.output_folder,
        input_columns=input_columns,
        target_columns=target_columns
    )
    training_node = train_component(
        input_folder = processing_node.outputs.output_folder,
        input_columns=input_columns,
        target_columns=target_columns
    )
    # testing_node = test_node(
    #     input_folder = processing_node.outputs.output_folder,
    #     input_pkl_folder = training_node.outputs.output_folder,
    # )

# Pipeline job with relevant inputs
pipeline_job = audi_setpoint_regressor(
    input_folder=input,
    input_columns = "drive_no,occWeight_R1L_IHAL_d_kg_G,occGender_R1L_IHAL_e_G,tOutsideTemp_IVAL_d_C_G,tCabinTemp_IVAL_d_C_G",
    target_columns = "tSetGblUsr_R1L_IHAL_d_C_G"
)

# Create or update the pipeline
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="audi setpoint pipeline"
)

# Console log
print(pipeline_job)