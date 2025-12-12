"""
Report creator module for generating and logging certification results.
Handles creation of filtered output files and CSV reports from results.
"""

import csv
import os

import comet_ml


def create_filtered_report(
    input_file: str,
    output_file: str,
    experiment: comet_ml.Experiment = None,
) -> None:
    """Create a filtered output file from certification results.

    Filters out misclassified instances (prediction == -2) from the original
    results file and creates a new file with only correctly classified predictions.

    Args:
        input_file: Path to the input results file
        output_file: Path to the output filtered results file
        experiment: Optional Comet ML experiment object for logging
    """
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line_idx, line in enumerate(f_in):
            if line_idx == 0:  # Keep header
                f_out.write(line)
            else:
                columns = line.strip().split("\t")
                if len(columns) >= 3:
                    prediction = columns[2]
                    # Skip rows where prediction is -2 (misclassified)
                    if prediction != "-2":
                        f_out.write(line)

    print(f"Filtered output file created: {output_file}")

    # Log to Comet ML if experiment is provided
    if experiment:
        try:
            experiment.log_asset(output_file, file_name=os.path.basename(output_file))
            print(f"Logged filtered report to Comet ML: {os.path.basename(output_file)}")
        except Exception as e:
            print(f"Warning: Failed to log filtered report to Comet ML: {e}")


def create_verona_csv(
    input_file: str,
    output_file: str,
    classifier_name: str,
    ddpm_model_name: str,
    sigma: float,
    alpha: float,
    N0: int,
    N: int,
    experiment: comet_ml.Experiment = None,
) -> None:
    """Create a VERONA-like result_df.csv from certification results.

    Converts the certification output file to VERONA format with the following fields:
    - network: classifier name
    - image_id: original CIFAR-10 index
    - original_label: true class label
    - tmp_path: placeholder
    - epsilon_value: certified robustness radius
    - smallest_sat_value: placeholder
    - total_time: certification time
    - verifier: description of the certification method and parameters

    Args:
        input_file: Path to the certify output results file (tab-separated)
        output_file: Path to the output VERONA CSV file
        classifier_name: Name/path of the classifier model
        ddpm_model_name: Name of the DDPM model used
        sigma: Noise level parameter
        alpha: Failure probability parameter
        N0: Number of noise samples for initial prediction
        N: Number of noise samples for certification
        experiment: Optional Comet ML experiment object for logging
    """
    verifier_string = (
        f"RS_{classifier_name}_"
        f"{ddpm_model_name}_{sigma}_{alpha}_{N0}_{N}"
    )

    with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
        csv_writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "network",
                "image_id",
                "original_label",
                "tmp_path",
                "epsilon_value",
                "smallest_sat_value",
                "total_time",
                "verifier",
            ],
        )
        csv_writer.writeheader()

        for line_idx, line in enumerate(f_in):
            if line_idx == 0:  # Skip header
                continue

            columns = line.strip().split("\t")
            if len(columns) >= 6:
                original_idx = columns[0]
                label = columns[1]
                radius = columns[3]
                time_elapsed = columns[5]

                csv_writer.writerow({
                    "network": verifier_string,
                    "image_id": original_idx,
                    "original_label": label,
                    "tmp_path": "n.a.",
                    "epsilon_value": radius,
                    "smallest_sat_value": "n.a.",
                    "total_time": time_elapsed,
                    "verifier": verifier_string,
                })

    print(f"VERONA-like CSV file created: {output_file}")

    # Log to Comet ML if experiment provided
    if experiment:
        try:
            experiment.log_asset(output_file, file_name=os.path.basename(output_file))
            print(f"Logged VERONA CSV to Comet ML: {os.path.basename(output_file)}")
        except Exception as e:
            print(f"Warning: Failed to log VERONA CSV to Comet ML: {e}")

