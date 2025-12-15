"""
Comet ML experiment tracking wrapper for certification experiments.
Handles all Comet ML operations with centralized error handling.
"""

import os
from typing import Any, Dict, Optional

import comet_ml


class CometTracker:
    """Wrapper for Comet ML experiment tracking."""

    def __init__(
        self,
        experiment_name: str,
        dataset_name: str,
        model_name: str,
        ddpm_model_name: str,
        project_name: str = "rs-rd",
        sigma: Optional[float] = None,
        alpha: Optional[float] = None,
        N0: Optional[int] = None,
        N: Optional[int] = None,
    ) -> None:
        """Initialize Comet ML experiment.

        Args:
            experiment_name: Name of the experiment
            dataset_name: Name of the dataset
            model_name: Name of the model
            ddpm_model_name: Name of the DDPM model
            project_name: Comet ML project name
            sigma: Noise hyperparameter for tagging
            alpha: Failure probability for tagging
            N0: Number of samples for initial prediction for tagging
            N: Number of samples for certification for tagging

        Returns:
            None (sets self.experiment to None if tracking fails)
        """
        self.experiment = None

        try:
            comet_ml.login()
            print("Comet ML login successful")
        except Exception as e:
            print(f"Warning: Comet ML login failed: {e}")
            return

        try:
            tags = ["certification", f"{dataset_name}", f"{model_name}", f"{ddpm_model_name}"]
            if sigma is not None:
                tags.append(f"sigma={sigma}")
            if alpha is not None:
                tags.append(f"alpha={alpha}")
            if N0 is not None:
                tags.append(f"N0={N0}")
            if N is not None:
                tags.append(f"N={N}")

            experiment_config = comet_ml.ExperimentConfig(
                name=experiment_name, tags=tags
            )
            self.experiment = comet_ml.start(
                project_name=project_name,
                experiment_config=experiment_config,
            )
            print(f"Comet ML experiment created: {self.experiment.url}")
        except Exception as e:
            print(f"Warning: Failed to create Comet ML experiment: {e}")
            print("Continuing without Comet ML tracking...")

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log experiment parameters.

        Args:
            parameters: Dictionary of parameters to log
        """
        if not self.experiment:
            return

        try:
            self.experiment.log_parameters(parameters)
        except Exception as e:
            print(f"Warning: Failed to log parameters to Comet ML: {e}")

    def log_metric(self, metric_name: str, value: float) -> None:
        """Log a single metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if not self.experiment:
            return

        try:
            self.experiment.log_metric(metric_name, value)
        except Exception as e:
            print(f"Warning: Failed to log metric '{metric_name}' to Comet ML: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step/epoch number
        """
        if not self.experiment:
            return

        try:
            self.experiment.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metrics to Comet ML: {e}")

    def log_other(self, key: str, value: Any) -> None:
        """Log arbitrary other data.

        Args:
            key: Key for the data
            value: Value to log
        """
        if not self.experiment:
            return

        try:
            self.experiment.log_other(key, value)
        except Exception as e:
            print(f"Warning: Failed to log '{key}' to Comet ML: {e}")

    def log_asset(self, file_path: str, file_name: Optional[str] = None) -> None:
        """Log a file asset.

        Args:
            file_path: Path to the file
            file_name: Optional custom name for the file
        """
        if not self.experiment:
            return

        try:
            self.experiment.log_asset(
                file_path, file_name=file_name or os.path.basename(file_path)
            )
        except Exception as e:
            print(f"Warning: Failed to log asset '{file_path}' to Comet ML: {e}")

    def end(self) -> None:
        """End the experiment."""
        if not self.experiment:
            return

        try:
            self.experiment.end()
            print(f"Experiment completed. View results at: {self.experiment.url}")
        except Exception as e:
            print(f"Warning: Failed to end Comet ML experiment: {e}")

    @property
    def is_active(self) -> bool:
        """Check if experiment tracking is active.

        Returns:
            True if experiment is active, False otherwise
        """
        return self.experiment is not None

