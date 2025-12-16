import comet_ml
import argparse
import datetime
import logging
import time

import numpy as np
import torch
from comet_tracker import CometTracker
from core import Smooth
from DRM import DiffusionRobustModel
from report_creator import create_filtered_report, create_verona_csv

#torch.manual_seed(0) #TODO ?
from utils import (
    UnflattenDataset,
    get_diffusion_model_path_name_tuple,
    override_args_with_cli,
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


def main(args=None):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---------------------------------------BASIC EXPERIMENT CONFIGURATION -----------------------------------------
    experiment_type = "certification"
    dataset_name = "CIFAR-10"
    split = "test"
    sample_size = 100
    random_seed = 42

    sample_correct_predictions = True
    sample_stratified = True
    # ----------------------------------------RANDOMIZED SMOOTHING VARIABLES------------------------------------------
    sigma = 0.25
    N0 = 100
    N = 100000
    batch_size = 1000
    alpha = 0.001
    
    # classifier_type = "huggingface"
    # classifier_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
    classifier_type = "pytorch"
    classifier_name = "conv_big_best"
    

    default_params = {
        "dataset_name": dataset_name,
        "sigma": sigma,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "N0": N0,
        "N": N,
        "batch_size": batch_size,
        "alpha": alpha,
        "sample_correct_predictions": sample_correct_predictions,
        "sample_stratified": sample_stratified,
        "classifier_type": classifier_type,
        "classifier_name": classifier_name,
    }
    params = override_args_with_cli(default_params, args)
    (dataset_name, sigma, sample_size, random_seed, N0, N, batch_size, alpha,
     sample_correct_predictions, sample_stratified, classifier_type, classifier_name) = params.values()

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

    # ----------------------------------------DATASET AND MODELS DIRECTORY CONFIGURATION---------------------------
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
    tracker.log_metric("experiment_start_time", start_time)
    
    # Initialize the model with specified classifier
    model = DiffusionRobustModel(
        classifier_type=classifier_type,
        classifier_name=classifier_name,
        models_dir=MODELS_DIR,
        dataset_name=dataset_name,
        device=device,
        image_size=image_size
    )

    sample_func = get_balanced_sample if sample_stratified else get_sample
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

    # ----------------------------------------DEFINE SMOOTHED CLASSIFIER --------------------------------------------
    smoothed_classifier = Smooth(model, num_classes, sigma, t, sample_correct_predictions=sample_correct_predictions)

    # ----------------------------------------CERTIFICATION PROCESS ------------------------------------------------
    f = open(str(output_file), 'w')
    print("original_idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    total_num = 0
    correct = 0
    total_samples = len(dataset)

    print(f"Starting certification on {total_samples} samples (seed={random_seed})")
    
    for i in range(len(dataset)):
        original_idx = original_indices[i]  # Get true index
        (x, label) = dataset[i]
        x = x.to(device)

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
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Log final metrics to Comet 
    tracker.log_metrics({
        "final_accuracy": final_accuracy,
        "total_samples_processed": total_num,
        "total_correct": correct,
        "experiment_start_time": start_time,
        "experiment_end_time": end_time,
        "experiment_duration_seconds": total_duration
    })
    
    print(f"Total experiment duration: {total_duration:.2f} seconds ({datetime.timedelta(seconds=int(total_duration))})")

    # Log experiment summary
    tracker.log_other("experiment_summary", {
        "total_balanced_samples": len(dataset),
        "split": split,
        "samples_processed": total_num,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "sampling_method": "Stratified" if sample_stratified else "Non-stratified",
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
    # 
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, default=None, help="noise hyperparameter")
    parser.add_argument("--sample_size", type=int, default=None, help="number of balanced samples to certify")
    parser.add_argument("--random_seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--N0", type=int, default=None, help="number of samples to use")
    parser.add_argument("--N", type=int, default=None, help="number of samples to use")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--alpha", type=float, default=None, help="failure probability")
    parser.add_argument("--outfile", type=str, default=None, help="output file")
    parser.add_argument(
        "--sample_correct_predictions",
        type=lambda x: x if isinstance(x, bool) else x.lower() in ("true", "1", "yes", "t"),
        default=None,
        help="only certify correctly classified samples",
    )
    parser.add_argument(
        "--sample_stratified",
        type=lambda x: x if isinstance(x, bool) else x.lower() in ("true", "1", "yes", "t"),
        default=None,
        help="use stratified sampling",
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        default=None,
        choices=["huggingface", "onnx", "pytorch"],
        help="Type of classifier to use"
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default=None,
        help="Name of the classifier model (HuggingFace model ID, ONNX model name, or PyTorch model name without .pth extension)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset (e.g., 'CIFAR-10', 'MNIST', 'ImageNet')"
    )

    args = parser.parse_args()
    main(args)