"""
Report creator module for generating and logging certification results.
Handles creation of filtered output files and CSV reports from results.
"""
from pathlib import Path
import numpy as np
import pandas as pd


class CSVResultWriter:
    """Manages CSV file writing for certification results with dynamic appending."""

    CSV_COLUMNS = [
        "network",
        "image_id",
        "original_label",
        "predicted_class",
        "tmp_path",
        "epsilon_value",
        "smallest_sat_value",
        "total_time",
        "verifier",
    ]

    def __init__(
        self,
        result_df_path: Path,
        misclassified_df_path: Path,
        abstained_df_path: Path,
        all_results_df_path: Path,
        summary_df_path: Path,
        verifier_string: str,
    ):
        """Initialize CSV result writer.

        Args:
            result_df_path: Path for main results CSV
            misclassified_df_path: Path for misclassified instances CSV
            abstained_df_path: Path for abstained instances CSV
            all_results_df_path: Path for combined results CSV (all instances)
            summary_df_path: Path for summary statistics CSV
            verifier_string: Verifier description string
        """
        self.result_df_path = result_df_path
        self.misclassified_df_path = misclassified_df_path
        self.abstained_df_path = abstained_df_path
        self.all_results_df_path = all_results_df_path
        self.summary_df_path = summary_df_path
        self.verifier_string = verifier_string

        # Initialize CSV files with headers
        pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(result_df_path, index=False)
        pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(misclassified_df_path, index=False)
        pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(abstained_df_path, index=False)
        pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(all_results_df_path, index=False)

        # Track statistics for summary
        self.certified_radii = []

    def append_result(
        self,
        image_id: int,
        original_label: int,
        predicted_class: int,
        epsilon_value: float,
        total_time: str,
        prediction_status: int,  # Smooth.MISCLASSIFIED, Smooth.ABSTAIN, or valid class
    ) -> None:
        """Append a single result to the appropriate CSV file.

        Args:
            image_id: Original dataset index
            original_label: True label
            predicted_class: Predicted class
            epsilon_value: Certified radius
            total_time: Time elapsed as string
            prediction_status: Status code (MISCLASSIFIED=-2, ABSTAIN=-1, or valid class)
        """
        row = {
            "network": self.verifier_string,
            "image_id": image_id,
            "original_label": original_label,
            "predicted_class": predicted_class,
            "tmp_path": np.nan,
            "epsilon_value": epsilon_value,
            "smallest_sat_value": np.nan,
            "total_time": total_time,
            "verifier": self.verifier_string,
        }

        df = pd.DataFrame([row])

        # Always append to all_results_df (combined file)
        df.to_csv(self.all_results_df_path, mode="a", header=False, index=False)

        # Also append to specific file based on status
        if prediction_status == -2:  # MISCLASSIFIED
            df.to_csv(self.misclassified_df_path, mode="a", header=False, index=False)
        elif prediction_status == -1:  # ABSTAIN
            df.to_csv(self.abstained_df_path, mode="a", header=False, index=False)
        else:  # Successful certification
            self.certified_radii.append(epsilon_value)
            df.to_csv(self.result_df_path, mode="a", header=False, index=False)

    def create_summary(
        self,
        total_num: int,
        correct: int,
        n_misclassified: int,
        n_abstain: int,
        sigma: float,
        alpha: float,
        N0: int,
        N: int,
        model_name: str,
        total_duration: float = 0.0,
        sum_certification_time: float = 0.0,
    ) -> None:
        """Create and save summary dataframe.

        Args:
            total_num: Total number of samples processed
            correct: Number of correct predictions
            n_misclassified: Number of misclassified instances
            n_abstain: Number of abstained instances
            sigma: Noise level parameter
            alpha: Failure probability parameter
            N0: Number of noise samples for initial prediction
            N: Number of noise samples for certification
            model_name: Classifier model name
            total_duration: Total experiment duration in seconds
            sum_certification_time: Sum of all individual certification times in seconds
        """
        # Overall accuracy accounting for misclassifications
        overall_accuracy = correct / float(total_num) if total_num > 0 else 0.0

        # Percentage of correctly classified instances that weren't abstained
        n_non_abstained = total_num - n_abstain
        accuracy_without_abstain = correct / float(n_non_abstained) if n_non_abstained > 0 else 0.0

        # Average certified radius (only for successfully certified instances)
        avg_certified_radius = np.mean(self.certified_radii) if self.certified_radii else 0.0

        summary_data = {
            "overall_accuracy": [overall_accuracy],
            "accuracy_without_abstain": [accuracy_without_abstain],
            "sigma": [sigma],
            "alpha": [alpha],
            "N0": [N0],
            "N": [N],
            "avg_certified_radius": [avg_certified_radius],
            "model_name": [model_name],
            "n_total": [total_num],
            "n_correct": [correct],
            "n_misclassified": [n_misclassified],
            "n_abstain": [n_abstain],
            "total_duration_seconds": [total_duration],
            "sum_certification_time_seconds": [sum_certification_time],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.summary_df_path, index=False)
        print(f"Saved summary dataframe to {self.summary_df_path}")

    def log_to_comet(self, tracker) -> None:
        """Log all CSV files to Comet ML.

        Args:
            tracker: CometTracker instance
        """
        tracker.log_asset(str(self.result_df_path))
        print(f"Result dataframe logged to Comet ML: {self.result_df_path}")
        tracker.log_asset(str(self.misclassified_df_path))
        print(f"Misclassified dataframe logged to Comet ML: {self.misclassified_df_path}")
        tracker.log_asset(str(self.abstained_df_path))
        print(f"Abstained dataframe logged to Comet ML: {self.abstained_df_path}")
        tracker.log_asset(str(self.all_results_df_path))
        print(f"All results dataframe logged to Comet ML: {self.all_results_df_path}")
        tracker.log_asset(str(self.summary_df_path))
        print(f"Summary dataframe logged to Comet ML: {self.summary_df_path}")

