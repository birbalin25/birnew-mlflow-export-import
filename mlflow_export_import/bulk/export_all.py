"""
Export the entire tracking server - all registerered models, experiments, runs and the Databricks notebook associated with the run.
"""

import os
import time
import click
import mlflow

from mlflow_export_import.common.click_options import (
    opt_output_dir,
    opt_export_latest_versions,
    opt_stages,
    opt_export_permissions,
    opt_run_start_time,
    opt_until,
    opt_export_deleted_runs,
    opt_export_version_model,
    opt_notebook_formats,
    opt_use_threads,
)
from mlflow_export_import.common.iterators import SearchExperimentsIterator
from mlflow_export_import.common import utils, io_utils
from mlflow_export_import.client.client_utils import create_mlflow_client
from mlflow_export_import.bulk.export_models import export_models
from mlflow_export_import.bulk.export_experiments import export_experiments
from mlflow_export_import.bulk import bulk_utils
from mlflow_export_import.bulk.model_utils import get_experiments_name_of_models
from mlflow_export_import.common.checkpoint_thread import CheckpointThread, filter_unprocessed_objects
from mlflow_export_import.bulk.model_utils import get_experiment_runs_dict_from_names
from mlflow_export_import.bulk.export_prompts import export_prompts
from mlflow_export_import.bulk.export_evaluation_datasets import export_evaluation_datasets

ALL_STAGES = "Production,Staging,Archived,None"

_logger = utils.getLogger(__name__)


def export_all(
        output_dir,
        run_start_time = None,
        runs_until = None,
        stages = "",
        export_latest_versions = False,
        export_deleted_runs = False,
        export_version_model = False,
        export_permissions = False,
        notebook_formats = None,
        use_threads  =  False,
        mlflow_client = None,
        task_index = None,
        num_tasks = None,
        checkpoint_dir_experiment = None,
        checkpoint_dir_model = None,
        model_names = None
    ):
    mlflow_client = mlflow_client or create_mlflow_client()

    ###
    ###


    start_time = time.time()
    res_models = export_models(
        mlflow_client = mlflow_client,
        # model_names = "all",
        model_names = model_names,
        output_dir = output_dir,
        stages = stages,
        export_latest_versions = export_latest_versions,
        export_all_runs = True,
        export_deleted_runs = export_deleted_runs,
        export_permissions = export_permissions,
        run_start_time = run_start_time,
        runs_until = runs_until,
        export_version_model = export_version_model,
        notebook_formats = notebook_formats,
        use_threads = use_threads,
        task_index = task_index,
        num_tasks = num_tasks,
        checkpoint_dir_experiment = checkpoint_dir_experiment,
        checkpoint_dir_model = checkpoint_dir_model

    )

    res_exps = None
    if not model_names.endswith('.txt'):
        all_exps = SearchExperimentsIterator(mlflow_client)
        all_exps = list(set(all_exps))
        all_exp_names = [ exp.name for exp in all_exps ]
        _logger.info(f"TOTAL WORKSPACE EXPERIMENT COUNT: {len(all_exp_names)}")

        all_model_exp_names=get_experiments_name_of_models(mlflow_client,model_names = "all")


        all_model_exp_names = list(set(all_model_exp_names))
        _logger.info(f"TOTAL WORKSPACE MODEL EXPERIMENT COUNT: {len(all_model_exp_names)}")

        remaining_exp_names = list(set(all_exp_names) - set(all_model_exp_names))
        _logger.info(f"TOTAL WORKSPACE EXPERIMENT COUNT WITH NO MODEL: {len(remaining_exp_names)}")


        remaining_exp_names_subset = bulk_utils.get_subset_list(remaining_exp_names, task_index, num_tasks)
        _logger.info(f"TOTAL WORKSPACE EXPERIMENT COUNT WITH NO MODEL FOR TASK_INDEX={task_index}:   {len(remaining_exp_names_subset)}")

        exps_and_runs = get_experiment_runs_dict_from_names(mlflow_client, remaining_exp_names_subset)

        exps_and_runs = filter_unprocessed_objects(checkpoint_dir_experiment,"experiments",exps_and_runs)    
        _logger.info(f"AFTER FILTERING OUT THE PROCESSED EXPERIMENTS FROM CHECKPOINT, TOTAL REMAINING COUNT: {len(exps_and_runs)}")

        res_exps = export_experiments(
            mlflow_client = mlflow_client,
            experiments = exps_and_runs,
            output_dir = os.path.join(output_dir,"experiments"),
            export_permissions = export_permissions,
            run_start_time = run_start_time,
            runs_until = runs_until,
            export_deleted_runs = export_deleted_runs,
            notebook_formats = notebook_formats,
            use_threads = use_threads,
            task_index = task_index,
            checkpoint_dir_experiment = checkpoint_dir_experiment
        )

    # Export prompts (returns dict with status)
    res_prompts = None
    try:
        _logger.info("Exporting prompts...")
        res_prompts = export_prompts(
            output_dir = os.path.join(output_dir, "prompts"),
            prompt_names = None,  # Export all prompts
            use_threads = use_threads,
            mlflow_client = mlflow_client
        )
        # Log if unsupported but don't fail
        if res_prompts and "unsupported" in res_prompts:
            _logger.warning(f"Prompts not supported in MLflow {res_prompts.get('mlflow_version')}")
        elif res_prompts and "error" in res_prompts:
            _logger.warning(f"Failed to export prompts: {res_prompts['error']}")
    except Exception as e:
        _logger.warning(f"Failed to export prompts: {e}")
        res_prompts = {"error": str(e)}

    # Export evaluation datasets (returns dict with status)
    res_datasets = None
    try:
        _logger.info("Exporting evaluation datasets...")
        res_datasets = export_evaluation_datasets(
            output_dir = os.path.join(output_dir, "evaluation_datasets"),
            dataset_names = None,  # Export all datasets
            experiment_ids = None,
            use_threads = use_threads,
            mlflow_client = mlflow_client
        )
        # Log if unsupported but don't fail
        if res_datasets and "unsupported" in res_datasets:
            _logger.warning(f"Evaluation datasets not supported in MLflow {res_datasets.get('mlflow_version')}")
        elif res_datasets and "error" in res_datasets:
            _logger.warning(f"Failed to export evaluation datasets: {res_datasets['error']}")
    except Exception as e:
        _logger.warning(f"Failed to export evaluation datasets: {e}")
        res_datasets = {"error": str(e)}

    duration = round(time.time() - start_time, 1)
    info_attr = {
        "options": {
            "stages": ALL_STAGES,
            "export_latest_versions": export_latest_versions,
            "export_permissions": export_permissions,
            "notebook_formats": notebook_formats,
            "use_threads": use_threads,
            "output_dir": output_dir,
        },
        "status": {
            "duration": duration,
            "models": res_models,
            "experiments": res_exps,
            "prompts": res_prompts,
            "evaluation_datasets": res_datasets
        }
    }
    io_utils.write_export_file(output_dir, "manifest.json", __file__, {}, info_attr)
    _logger.info(f"Duration for entire tracking server export: {duration} seconds")


@click.command()
@opt_output_dir
@opt_export_latest_versions
@opt_stages
@opt_run_start_time
@opt_until
@opt_export_deleted_runs
@opt_export_version_model
@opt_export_permissions
@opt_notebook_formats
@opt_use_threads

def main(output_dir, stages, export_latest_versions, run_start_time, runs_until,
        export_deleted_runs,
        export_version_model,
        export_permissions,
        notebook_formats, use_threads
     ):
    _logger.info("Options:")
    for k,v in locals().items():
        _logger.info(f"  {k}: {v}")
    export_all(
        output_dir = output_dir,
        stages = stages,
        run_start_time = run_start_time,
        runs_until = runs_until,
        export_latest_versions = export_latest_versions,
        export_deleted_runs = export_deleted_runs,
        export_version_model = export_version_model,
        export_permissions = export_permissions,
        notebook_formats = notebook_formats,
        use_threads = use_threads
    )


if __name__ == "__main__":
    main()
