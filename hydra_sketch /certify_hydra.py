"""
Hydra-enabled version of certify.py for parameter sweeps and SLURM job scheduling.

Usage:
    Single run: python certify_hydra.py
    Override: python certify_hydra.py sigma=0.5 sample_size=200
    Sweep (local): python certify_hydra.py -m sigma=0.25,0.5,0.75 sample_size=100,200
    Sweep (SLURM): python certify_hydra.py -m sigma=0.25,0.5,0.75 --config-name certify
"""

import datetime
import logging
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
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
    """Main function that receives Hydra config."""
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
    
    # Log experiment parameters
    tracker.log_parameters({
        "sigma": sigma,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "N0": N0,
        "N": N,
        "batch_size": batch_size,
        "alpha": alpha,
        "outfile": str(output_file),
        "dataset": dataset_name,
        "classifier_type": classifier_type,
        "classifier_name": classifier_name
    })
    
    # Log experiment start time
    start_time = time.time()
    tracker.log_metric("experiment_start_time", start_time)
    
    # Initialize the model with specified classifier
    model = DiffusionRobustModel(
        classifier_type=classifier_type,
        classifier_name=classifier_name,
        models_dir=MODELS_DIR,
        dataset_name=dataset_name
    )

    sample_func = get_balanced_sample if stratified else get_sample
    sampled_dataset, original_indices = sample_func(
        dataset_name=dataset_name,
        train_bool=(split == "train"),
        dataset_size=sample_size,
        dataset_dir=DATASET_DIR,
        seed=random_seed,
        image_size=None 
    )
    
    height, width = image_size[1], image_size[0]
    dataset = UnflattenDataset(sampled_dataset, channels=num_channels, height=height, width=width)
    
    indices_file = save_original_indices(
        dataset_name=dataset_name,
        original_indices=original_indices,
        output_dir=RESULTS_DIR,
        sample_size=sample_size,
        timestamp=timestamp,
        split=split,
    )

    tracker.log_asset(str(indices_file))
    
    # Get the timestep t corresponding to noise level sigma
    target_sigma = sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    # Define smoothed classifier
    smoothed_classifier = Smooth(model, num_classes, sigma, t, sample_correct_predictions=sample_correct_predictions)

    f = open(str(output_file), 'w')
    print("original_idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    total_num = 0
    correct = 0
    total_samples = len(dataset)

    print(f"Starting certification on {total_samples} samples (seed={random_seed})")
    
    for i in range(len(dataset)):
        original_idx = original_indices[i]  # Get true index
        (x, label) = dataset[i]
        x = x.cuda()

        before_time = time.time()
        prediction, radius = smoothed_classifier.certify(x, N0, N, alpha, batch_size, label=label)
        after_time = time.time()

        # Only count time for certification, not for misclassified samples
        certification_time = 0.0 if prediction == Smooth.MISCLASSIFIED else (after_time - before_time)

        correct += int(prediction == label)
        total_num += 1

        time_elapsed = str(datetime.timedelta(seconds=certification_time))
        current_accuracy = correct / float(total_num)

        tracker.log_metrics({
            "original_cifar10_index": original_idx,
            "subset_index": i,
            "prediction": prediction,
            "true_label": label,
            "radius": radius,
            "correct": correct,
            "total_processed": total_num,
            "current_accuracy": current_accuracy,
            "sample_correct_predictions": int(sample_correct_predictions),
            "certification_time_seconds": certification_time,
            "progress_percentage": (total_num / total_samples) * 100
        }, step=total_num)

        # Log individual sample results
        tracker.log_other(f"sample_{original_idx}_result", {
            "original_cifar10_index": original_idx,
            "subset_index": i,
            "true_label": label,
            "prediction": prediction,
            "radius": radius,
            "correct": prediction == label,
            "time_elapsed": time_elapsed
        })

        print(f"{original_idx}\t{label}\t{prediction}\t{radius:.3}\t{correct}\t{time_elapsed}", file=f, flush=True)
        
        # Print progress every 10 samples
        if total_num % 10 == 0:
            print(f"Progress: {total_num}/{total_samples} samples processed ({current_accuracy:.4f} accuracy)")

    final_accuracy = correct / float(total_num)
    print("sigma %.2f accuracy of smoothed classifier %.4f " % (sigma, final_accuracy))
    f.close()
    
    # Log final metrics to Comet 
    tracker.log_metrics({
        "final_accuracy": final_accuracy,
        "total_samples_processed": total_num,
        "total_correct": correct,
        "experiment_duration_seconds": time.time() - start_time
    })

    # Log experiment summary
    tracker.log_other("experiment_summary", {
        "total_balanced_samples": len(dataset),
        "split": split,
        "samples_processed": total_num,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "sampling_method": "Stratified" if stratified else "Non-stratified",
        "certification_parameters": {
            "N0": N0,
            "N": N,
            "alpha": alpha,
            "batch_size": batch_size
        },
        "output_file_path": str(output_file)
    })

    # Log output files to Comet
    tracker.log_asset(str(output_file))
    print(f"Original results file logged to Comet ML: {output_file}")
    
    if sample_correct_predictions:
        # Filter out misclassified instances and create a filtered output file
        filtered_output_file = output_file.with_name(output_file.stem + "_filtered.txt")
        create_filtered_report(str(output_file), str(filtered_output_file), tracker.experiment)
    else:
        filtered_output_file = None
    
    verona_input_file = filtered_output_file if sample_correct_predictions else output_file
    verona_csv_file = verona_input_file.with_name(verona_input_file.stem + "_result_df.csv")
    create_verona_csv(
        str(verona_input_file),
        str(verona_csv_file),
        classifier_name_short,
        ddpm_model_name,
        sigma,
        alpha,
        N0,
        N,
        tracker.experiment,
    )
    # End experiment
    if tracker.is_active:
        tracker.end()
    else:
        print("Experiment completed (Comet ML tracking not avail)")


if __name__ == "__main__":
    # Hydra entry point
    # Usage:
    #   Single run: python certify_hydra.py
    #   Override: python certify_hydra.py sigma=0.5 sample_size=200
    #   Sweep: python certify_hydra.py -m sigma=0.25,0.5,0.75 sample_size=100,200
    
    try:
        from hydra import main as hydra_main
        
        @hydra_main(version_base=None, config_path="conf", config_name="certify")
        def hydra_entry(cfg: DictConfig) -> None:
            """Hydra entry point - replaces argparse."""
            main(cfg)
        
        hydra_entry()
    except ImportError:
        print("Error: hydra-core not installed. Install with: pip install hydra-core hydra-submitit-launcher")
        print("Falling back to manual config loading...")
        # Fallback: manual config loading
        with initialize(config_path="conf", version_base=None):
            cfg = compose(config_name="certify")
            main(cfg)

