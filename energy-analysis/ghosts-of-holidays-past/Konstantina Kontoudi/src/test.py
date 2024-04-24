import pandas as pd
from rtdip_sdk.pipelines.execute import PipelineJob, PipelineStep, PipelineTask, PipelineJobExecute

from src.feature_extraction import prepare_dataset
from src.rtdip_sources import OpenMeteoHistoricalWeatherSource, MultiFileSource
from src.rtdip_destinations import DiskDestination
from src.rtdip_transformations import FunctionTransformer

from src.settings import DATA_DIR, LAT_COL, LON_COL
from src.training import train


def main():
    scrooge_electricity_path = DATA_DIR / "scrooge_bldg.parquet"
    scrooge_weather_path = DATA_DIR / "scrooge_weather.parquet"

    load_raw_data_step = PipelineStep(
        name="load_raw_data",
        description="Load raw data",
        component=MultiFileSource,
        component_parameters={
            "file_confs": [
                (str(scrooge_weather_path), "parquet"),
                (str(scrooge_electricity_path), "parquet")
            ],
        },
        provide_output_to_step=["feature_extraction"]
    )

    feature_extraction_step = PipelineStep(
        name="feature_extraction",
        description="Extract features",
        component=FunctionTransformer,
        component_parameters={
            "function": prepare_dataset,
        },
        provide_output_to_step=["dump_features"],
        depends_on_step=["load_raw_data"]
    )

    dump_features_step = PipelineStep(
        name="dump_features",
        description="Store weather data to disk",
        component=DiskDestination,
        component_parameters={
            "path": DATA_DIR / "features.parquet",
            "file_type": "parquet"
        },
        depends_on_step=["feature_extraction"]
    )

    training_step = PipelineStep(
        name="training",
        description="Train model",
        component=FunctionTransformer,
        component_parameters={
            "function": train,
        },
        provide_output_to_step=["dump_model"],
        depends_on_step=["feature_extraction"]
    )

    dump_model_step = PipelineStep(
        name="dump_model",
        description="Store model to disk",
        component=DiskDestination,
        component_parameters={
            "path": DATA_DIR / "models.pickle",
            "file_type": "pickle"
        },
        depends_on_step=["training"]
    )

    modeling_steps = [load_raw_data_step, feature_extraction_step, dump_features_step, training_step, dump_model_step]

    modeling_task = PipelineTask(
        name="modeling",
        description="Modeling",
        step_list=modeling_steps,
        batch_task=True
    )

    pipeline_job = PipelineJob(
        name="test_job",
        description="test_job",
        version="0.0.1",
        task_list=[modeling_task]
    )

    pipeline = PipelineJobExecute(pipeline_job)

    result = pipeline.run()


if __name__ == "__main__":
    main()
