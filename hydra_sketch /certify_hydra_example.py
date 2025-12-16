"""
Example of how to convert certify.py to use Hydra instead of argparse.
This is a reference implementation - not meant to replace certify.py directly.
"""

import datetime
import logging
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from comet_tracker import CometTracker
from core import Smooth
from DRM import DiffusionRobustModel
from report_creator import create_filtered_report, create_verona_csv

from utils import (
    UnflattenDataset,
    get_diffusion_model_path_name_tuple,
)

from ada_verona import (
    get_balanced_sample,
    get_dataset_config,
    get_dataset_dir,
    get_models_dir,
    get_results_dir,
    get_sample,
    save_original_indices,
    create_experiment_directory,
)

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("dulwich").setLevel(logging.WARNING)
logging.getLogger("comet_ml").setLevel(logging.INFO)


def main(cfg: DictConfig):
    """Main function that receives Hydra config instead of argparse args."""
    # Access config values directly
    dataset_name = cfg.dataset_name
    split = cfg.split
    sample_size = cfg.sample_size
    random_seed = cfg.random_seed
    sigma = cfg.sigma
    N0 = cfg.N0
    N = cfg.N
    batch_size = cfg.batch_size
    alpha = cfg.alpha
    sample_correct_predictions = cfg.sample_correct_predictions
    stratified = cfg.stratified
    classifier_type = cfg.classifier_type
    classifier_name = cfg.classifier_name
    experiment_type = cfg.experiment_type

    classifier_name_short = classifier_name.split("/")[-1] if classifier_name else "unknown"

    dataset_config_map = get_dataset_config()
    if dataset_name not in dataset_config_map:
        raise ValueError(
            f"Unsupported dataset: '{dataset_name}'. "
            f"Supported datasets: {', '.join(dataset_config_map.keys())}"
        )

    dataset_config = dataset_config_map[dataset_name]
    image_size = dataset_config["default_size"]
    num_channels = dataset_config["channels"]
    num_classes = dataset_config["num_classes"]
    
    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name)
    RESULTS_DIR = get_results_dir(dataset_name)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{classifier_name_short}_{sigma}_{dataset_name}_{timestamp}"
    _, ddpm_model_name = get_diffusion_model_path_name_tuple(dataset_name)

    tracker = CometTracker(
        experiment_name,
        dataset_name,
        classifier_name_short,
        ddpm_model_name,
        sigma=sigma,
        alpha=alpha,
        N0=N0,
        N=N,
    )
    experiment_folder = create_experiment_directory(
        results_dir=RESULTS_DIR,
        experiment_type=experiment_type,
        dataset_name=dataset_name,
        timestamp=timestamp,
    )

    output_file = experiment_folder / f"{experiment_name}_{timestamp}.txt"
    
    # Rest of the function would be the same...
    # (omitted for brevity)
    print(f"Config: {OmegaConf.to_yaml(cfg)}")


# To use Hydra decorator, uncomment and install hydra-core:
# from hydra import main as hydra_main_decorator
# 
# @hydra_main_decorator(version_base=None, config_path="conf", config_name="certify")
# def hydra_main(cfg: DictConfig) -> None:
#     """Hydra entry point - replaces argparse."""
#     main(cfg)


if __name__ == "__main__":
    # Option 1: Use Hydra decorator (requires conf/certify.yaml)
    # hydra_main()
    
    # Option 2: Use without decorator for programmatic access
    # This allows you to call main() directly with a config dict
    from omegaconf import DictConfig
    
    # Example: Create config programmatically or from YAML
    default_config = {
        "dataset_name": "CIFAR-10",
        "split": "test",
        "sample_size": 100,
        "random_seed": 42,
        "sigma": 0.25,
        "N0": 100,
        "N": 100000,
        "batch_size": 1000,
        "alpha": 0.001,
        "sample_correct_predictions": True,
        "stratified": True,
        "classifier_type": "huggingface",
        "classifier_name": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        "experiment_type": "certification",
    }
    
    cfg = OmegaConf.create(default_config)
    main(cfg)

