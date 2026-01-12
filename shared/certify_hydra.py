"""Unified Hydra-enabled certification script for randomized smoothing experiments."""

import comet_ml
import datetime
import logging
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from shared.tracking.comet_tracker import CometTracker
from shared.io.csv_result_writer import CSVResultWriter
from shared.io.signal_handler import setup_signal_handler
from shared.utils.utils import (
    get_diffusion_model_path_name_tuple,
    get_device_with_diagnostics,
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
    OmegaConf.set_struct(cfg, False)
    
    start_time = time.time()
    
    device = get_device_with_diagnostics()

    experiment_type = cfg.get("experiment_type", "certification")
    dataset_name = cfg.dataset_name
    split = cfg.split
    sample_size = cfg.sample_size
    random_seed = cfg.random_seed
    sigma = cfg.sigma
    N0 = cfg.N0
    N = cfg.N
    batch_size = cfg.batch_size
    alpha = cfg.alpha
    sample_correct_predictions = cfg.get("sample_correct_predictions", True)
    sample_stratified = cfg.get("sample_stratified", cfg.get("stratified", False))
    mode = cfg.get("mode", "certify")
    if mode not in ["certify", "predict", "base_predict"]:
        raise ValueError(f"Invalid mode: '{mode}'. Must be one of: 'certify', 'predict', 'base_predict'")
    classifier_type = cfg.classifier_type
    classifier_name = cfg.classifier_name

    classifier_name_short = classifier_name.split("/")[-1] if classifier_name else "unknown"

    dataset_config_map = get_dataset_config()
    if dataset_name not in dataset_config_map:
        raise ValueError(
            f"Unsupported dataset: '{dataset_name}'. "
            f"Supported datasets: {', '.join(dataset_config_map.keys())}"
        )

    dataset_config = dataset_config_map[dataset_name]
    image_size = dataset_config["default_size"]
    num_classes = dataset_config["num_classes"]

    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name)
    RESULTS_DIR = get_results_dir(dataset_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = f"{classifier_name_short}_{sigma}_{dataset_name}_{timestamp}"
    _, ddpm_model_name = get_diffusion_model_path_name_tuple(dataset_name)

    verifier_string = (
        f"RS_{classifier_name_short}_"
        f"{ddpm_model_name}_{sigma}_{alpha}_{N0}_{N}"
    )

    experiment_tag = cfg.get("experiment_tag", None)
    tracker = CometTracker(
        experiment_name,
        dataset_name,
        classifier_name_short,
        ddpm_model_name,
        sigma=sigma,
        alpha=alpha,
        N0=N0,
        N=N,
        experiment_tag=experiment_tag,
    )
    experiment_folder = create_experiment_directory(
        results_dir=RESULTS_DIR,
        experiment_type=experiment_type,
        dataset_name=dataset_name,
        timestamp=timestamp,
        classifier_name=classifier_name_short,
        experiment_tag=experiment_tag,
    )

    # Output files
    result_df_path = experiment_folder / "result_df.csv"
    misclassified_df_path = experiment_folder / "misclassified_df.csv"
    abstained_df_path = experiment_folder / "abstained_df.csv"
    all_results_df_path = experiment_folder / "all_results_df.csv"
    summary_df_path = experiment_folder / "summary_df.csv"
    output_file = experiment_folder / f"{experiment_name}.txt"

    csv_writer = CSVResultWriter(
        result_df_path=result_df_path,
        misclassified_df_path=misclassified_df_path,
        abstained_df_path=abstained_df_path,
        all_results_df_path=all_results_df_path,
        summary_df_path=summary_df_path,
        verifier_string=verifier_string,
    )

    tracker.log_parameters({
        "sigma": sigma,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "N0": N0,
        "N": N,
        "batch_size": batch_size,
        "alpha": alpha,
        "dataset": dataset_name,
        "classifier_type": classifier_type,
        "classifier_name": classifier_name,
        "mode": mode,
    })

    tracker.log_metric("experiment_start_time", start_time)

    # Import dataset-specific DRM modules
    from shared.core import Smooth
    
    if dataset_name == "CIFAR-10":
        from ds_cifar10.DRM import DiffusionRobustModel
    elif dataset_name == "ImageNet":
        from ds_imagenet.DRM import DiffusionRobustModel
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize model with specified classifier
    model = DiffusionRobustModel(
        classifier_type=classifier_type,
        classifier_name=classifier_name,
        models_dir=MODELS_DIR,
        dataset_name=dataset_name,
        device=device,
        image_size=image_size
    )

    sample_func = get_balanced_sample if sample_stratified else get_sample
    dataset, original_indices = sample_func(
        dataset_name=dataset_name,
        train_bool=(split == "train"),
        dataset_size=sample_size,
        dataset_dir=DATASET_DIR,
        seed=random_seed,
        image_size=None,
        flatten=False,
    )

    indices_file = save_original_indices(
        dataset_name=dataset_name,
        original_indices=original_indices,
        output_dir=experiment_folder,
        sample_size=sample_size,
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

    # Certification process
    total_num = 0
    correct = 0
    n_misclassified = 0
    n_abstain = 0
    sum_certification_time = 0.0
    total_samples = len(dataset)

    def get_summary_params():
        """Get current summary parameters for signal handler."""
        current_time = time.time()
        current_total_duration = current_time - start_time
        return {
            "total_num": total_num,
            "correct": correct,
            "n_misclassified": n_misclassified,
            "n_abstain": n_abstain,
            "sigma": sigma,
            "alpha": alpha,
            "N0": N0,
            "N": N,
            "model_name": classifier_name_short,
            "total_duration": current_total_duration,
            "sum_certification_time": sum_certification_time,
        }

    setup_signal_handler(csv_writer, tracker, output_file, get_summary_params)

    mode_str_map = {
        "certify": "certification",
        "predict": "prediction",
        "base_predict": "base prediction"
    }
    mode_str = mode_str_map.get(mode, mode)
    print(f"Starting {mode_str} on {total_samples} samples (seed={random_seed})")

    f = open(str(output_file), 'w')
    print("original_idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # Dynamic field name for Comet logging based on dataset
    index_field_name = f"original_{dataset_name.lower().replace('-', '_')}_index"
    
    try:
        for i in range(len(dataset)):
            original_idx = original_indices[i]
            (x, label) = dataset[i]
            x = x.to(device)
            
            before_time = time.time()
            if mode == "base_predict":
                prediction = smoothed_classifier.base_predict(x)
                radius = 0.0
            elif mode == "predict":
                prediction = smoothed_classifier.predict(x, N, alpha, batch_size)
                radius = 0.0
            else:  # mode == "certify"
                prediction, radius = smoothed_classifier.certify(x, N0, N, alpha, batch_size, label=label)
            after_time = time.time()
            
            certification_time = 0.0 if (prediction == Smooth.MISCLASSIFIED) else (after_time - before_time)
            sum_certification_time += certification_time

            correct += int(prediction == label)
            total_num += 1

            time_elapsed = str(datetime.timedelta(seconds=certification_time))
            current_accuracy = correct / float(total_num)

            if mode == "certify":
                if prediction == Smooth.MISCLASSIFIED:
                    n_misclassified += 1
                elif prediction == Smooth.ABSTAIN:
                    n_abstain += 1
            elif mode == "predict":
                if prediction == Smooth.ABSTAIN:
                    n_abstain += 1
                elif prediction != label:
                    n_misclassified += 1

            csv_writer.append_result(
                image_id=original_idx,
                original_label=label,
                predicted_class=prediction,
                epsilon_value=radius,
                total_time=time_elapsed,
                prediction_status=prediction,
            )

            tracker.log_metrics({
                index_field_name: original_idx,
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

            tracker.log_other(f"sample_{original_idx}_result", {
                index_field_name: original_idx,
                "subset_index": i,
                "true_label": label,
                "prediction": prediction,
                "radius": radius,
                "correct": prediction == label,
                "time_elapsed": time_elapsed
            })

            print(f"{original_idx}\t{label}\t{prediction}\t{radius:.3}\t{correct}\t{time_elapsed}", file=f, flush=True)

            if total_num % 20 == 0:
                print(f"Progress: {total_num}/{total_samples} samples processed ({current_accuracy:.4f} accuracy)")

            if total_num % 50 == 0:
                try:
                    # Update summary periodically to ensure it exists even if process is killed
                    current_time = time.time()
                    current_total_duration = current_time - start_time
                    csv_writer.create_summary(
                        total_num=total_num,
                        correct=correct,
                        n_misclassified=n_misclassified,
                        n_abstain=n_abstain,
                        sigma=sigma,
                        alpha=alpha,
                        N0=N0,
                        N=N,
                        model_name=classifier_name_short,
                        total_duration=current_total_duration,
                        sum_certification_time=sum_certification_time,
                    )
                    csv_writer.log_to_comet(tracker)
                    tracker.log_asset(str(output_file))
                    print(f"Periodic backup: CSV files and txt file logged to Comet ML (sample {total_num})")
                except Exception as e:
                    print(f"Warning: Failed to log CSV files to Comet ML: {e}")
    finally:
        # Ensure file is closed and summary is created even if loop is interrupted
        f.close()
        
        # Create/update summary with current progress if not already done
        if total_num > 0:
            current_time = time.time()
            current_total_duration = current_time - start_time
            try:
                csv_writer.create_summary(
                    total_num=total_num,
                    correct=correct,
                    n_misclassified=n_misclassified,
                    n_abstain=n_abstain,
                    sigma=sigma,
                    alpha=alpha,
                    N0=N0,
                    N=N,
                    model_name=classifier_name_short,
                    total_duration=current_total_duration,
                    sum_certification_time=sum_certification_time,
                )
            except Exception as e:
                print(f"Warning: Failed to create summary in finally block: {e}")

    final_accuracy = correct / float(total_num) if total_num > 0 else 0.0
    if mode == "base_predict":
        print("clean accuracy of base classifier %.4f " % final_accuracy)
    elif mode == "predict":
        print("sigma %.2f accuracy of smoothed classifier (predict mode) %.4f " % (sigma, final_accuracy))
    else:  # mode == "certify"
        print("sigma %.2f accuracy of smoothed classifier %.4f " % (sigma, final_accuracy))

    end_time = time.time()
    total_duration = end_time - start_time

    csv_writer.create_summary(
        total_num=total_num,
        correct=correct,
        n_misclassified=n_misclassified,
        n_abstain=n_abstain,
        sigma=sigma,
        alpha=alpha,
        N0=N0,
        N=N,
        model_name=classifier_name_short,
        total_duration=total_duration,
        sum_certification_time=sum_certification_time,
    )

    tracker.log_metrics({
        "final_accuracy": final_accuracy,
        "total_samples_processed": total_num,
        "total_correct": correct,
        "experiment_start_time": start_time,
        "experiment_end_time": end_time,
        "experiment_duration_seconds": total_duration
    })

    print(f"Total experiment duration: {total_duration:.2f} seconds ({datetime.timedelta(seconds=int(total_duration))})")

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
    })

    csv_writer.log_to_comet(tracker)
    tracker.log_asset(str(output_file))
    print(f"Results txt file logged to Comet ML: {output_file}")
    if tracker.is_active:
        tracker.end()
    else:
        print("Experiment completed (Comet ML tracking not avail)")

