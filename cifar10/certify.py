import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
#torch.manual_seed(0) #TODO ?
from torchvision import transforms, datasets

# Add parent directory to path to import rs_rd_research_code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import Smooth
from DRM import DiffusionRobustModel
from rs_rd_research.paths import get_dataset_dir, get_models_dir, get_results_dir
from utils import (
    create_experiment_folder,
    get_balanced_sample,
    get_sample,
    get_diffusion_model_path_name_tuple,
)
from report_creator import create_filtered_report, create_verona_csv
from comet_tracker import CometTracker

DATASET_DIR = get_dataset_dir()
MODELS_DIR = get_models_dir()
RESULTS_DIR = get_results_dir()


def main(args):
    # Create a descriptive experiment name that includes classifier_name and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use the classifier_name parameter
    classifier_name_short = args.classifier_name.split("/")[-1] if args.classifier_name else "unknown"
    safe_classifier_name = classifier_name_short.replace("-", "_")
    # Extract dataset name from the path
    dataset_name = os.path.basename(DATASET_DIR)
    experiment_name = f"{classifier_name_short}_{args.sigma}_{dataset_name}_{timestamp}"
    _, ddpm_model_name = get_diffusion_model_path_name_tuple(dataset_name)
    # Initialize Comet ML tracker
    tracker = CometTracker(
        experiment_name,
        dataset_name,
        classifier_name_short,
        ddpm_model_name,
        sigma=args.sigma,
        alpha=args.alpha,
        N0=args.N0,
        N=args.N,
    )

    # Create experiment folder for this configuration
    experiment_folder = create_experiment_folder(
        RESULTS_DIR,
        safe_classifier_name,
        args.sigma,
        args.alpha,
        args.N0,
        args.N,
        args.batch_size,
    )

    output_file = os.path.join(
        experiment_folder, f"{experiment_name}_{timestamp}.txt"
    )
    
    # Log experiment parameters
    tracker.log_parameters({
        "sigma": args.sigma,
        "sample_size": args.sample_size,
        "random_seed": args.random_seed,
        "N0": args.N0,
        "N": args.N,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "outfile": output_file,
        "dataset": "CIFAR-10",
        "classifier_type": args.classifier_type,
        "classifier_name": args.classifier_name
    })
    
    # Log experiment start time
    start_time = time.time()
    tracker.log_metric("experiment_start_time", start_time)
    
    # Initialize the model with specified classifier
    model = DiffusionRobustModel(
        classifier_type=args.classifier_type,
        classifier_name=args.classifier_name,
        models_dir=MODELS_DIR,
        dataset_name=dataset_name
    )

    # CIFAR-10 specific #TODO: change this according to the dataset
    full_dataset = datasets.CIFAR10(DATASET_DIR, train=False, download=False, transform=transforms.ToTensor())
    
    # Get samples using stratified or random sampling
    if args.stratified:
        dataset, original_indices = get_balanced_sample(
            full_dataset,
            train_bool=False,
            seed=args.random_seed,
            sample_size=args.sample_size
        )
    else:
        dataset, original_indices = get_sample(
            full_dataset,
            seed=args.random_seed,
            sample_size=args.sample_size
        )
    
    # Save original CIFAR-10 indices
    indices_file = os.path.join(RESULTS_DIR, f"original_cifar10_indices_nsample_{args.sample_size}_{timestamp}.txt")
    np.savetxt(indices_file, original_indices, fmt='%d',
               header=f"Original CIFAR-10 test indices for balanced sample (n_sample={args.sample_size})")

    # Log indices file to Comet ML
    tracker.log_asset(indices_file)
    
    # Get the timestep t corresponding to noise level sigma
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    # Define the smoothed classifier
    smoothed_classifier = Smooth(model, 10, args.sigma, t, sample_correct_predictions=args.sample_correct_predictions)

    f = open(output_file, 'w')
    print("original_idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    total_num = 0
    correct = 0
    total_samples = len(dataset)

    print(f"Starting certification on {total_samples} balanced samples (seed={args.random_seed})...")
    
    for i in range(len(dataset)):
        original_idx = original_indices[i]  # Get true CIFAR-10 index
        (x, label) = dataset[i]
        x = x.cuda()

        before_time = time.time()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size, label=label)
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
            "sample_correct_predictions": int(args.sample_correct_predictions),
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

        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            original_idx, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        
        # Print progress every 10 samples
        if total_num % 10 == 0:
            print(f"Progress: {total_num}/{total_samples} samples processed ({current_accuracy:.4f} accuracy)")

    final_accuracy = correct / float(total_num)
    print("sigma %.2f accuracy of smoothed classifier %.4f " % (args.sigma, final_accuracy))
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
        "samples_processed": total_num,
        "sample_size": args.sample_size,
        "random_seed": args.random_seed,
        "sampling_method": "Statified" if args.stratified else "Non-stratified",
        "certification_parameters": {
            "N0": args.N0,
            "N": args.N,
            "alpha": args.alpha,
            "batch_size": args.batch_size
        },
        "output_file_path": output_file
    })

    # Log output files to Comet
    tracker.log_asset(output_file)
    print(f"Original results file logged to Comet ML: {output_file}")
    
    if args.sample_correct_predictions:
        # Filter out misclassified instances and create a filtered output file
        filtered_output_file = output_file.replace(".txt", "_filtered.txt")
        create_filtered_report(output_file, filtered_output_file, tracker.experiment)
    
    
    verona_input_file = filtered_output_file if args.sample_correct_predictions else output_file
    verona_csv_file = verona_input_file.replace(".txt", "_result_df.csv")
    create_verona_csv(
        verona_input_file,
        verona_csv_file,
        classifier_name_short,
        ddpm_model_name,
        args.sigma,
        args.alpha,
        args.N0,
        args.N,
        tracker.experiment,
    )
    # End experiment
    if tracker.is_active:
        tracker.end()
    else:
        print("Experiment completed (Comet ML tracking was not available)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--sample_size", type=int, default=100, help="number of balanced samples to certify (default: 100)")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility (default: 42)")
    parser.add_argument("--N0", type=int, default=100, help="number of samples to use")
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--outfile", type=str, help="output file")
    parser.add_argument("--sample_correct_predictions", type=bool, default=True, help="only certify correctly classified samples (default: True)")
    parser.add_argument("--stratified", type=bool, default=True, help="use stratified sampling (default: True)")
    # Classifier selection arguments
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="huggingface",
        choices=["huggingface", "onnx"],
        help="Type of classifier to use"
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="Name of the classifier model (HuggingFace model ID or ONNX model name)"
    )

    args = parser.parse_args()
    main(args)