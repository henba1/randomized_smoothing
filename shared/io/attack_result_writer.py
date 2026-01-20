"""
CSV writer for adversarial attack runs (e.g., PGD-EOT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class AttackCSVWriter:
    CSV_COLUMNS = [
        "network",
        "image_id",
        "true_label",
        "cert_status",
        "cert_pred",
        "cert_radius_l2",
        "adv_status",
        "adv_pred",
        "within_cert_ball",
        "success_within_cert",
        "min_adv_radius_l2",
        "attack",
        "attack_eps_l2",
        "step_size",
        "num_iter",
        "eot_samples",
        "linf",
        "l2",
        "total_time",
        "artifact_path",
    ]

    def __init__(
        self,
        results_df_path: Path,
        all_results_df_path: Path,
        summary_df_path: Path,
        verifier_string: str,
    ) -> None:
        self.results_df_path = results_df_path
        self.all_results_df_path = all_results_df_path
        self.summary_df_path = summary_df_path
        self.verifier_string = verifier_string

        pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(self.results_df_path, index=False)
        pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(self.all_results_df_path, index=False)

        self._success_flags: list[bool] = []
        self._linf_values: list[float] = []
        self._l2_values: list[float] = []

    def append_result(
        self,
        *,
        image_id: int,
        true_label: int,
        cert_status: int,
        cert_pred: int,
        cert_radius_l2: float,
        adv_status: int,
        adv_pred: int,
        within_cert_ball: bool,
        success_within_cert: bool,
        min_adv_radius_l2: float | None,
        attack_name: str,
        attack_eps_l2: float,
        step_size: float,
        num_iter: int,
        eot_samples: int,
        linf: float,
        l2: float,
        total_time: str,
        artifact_path: str | None,
    ) -> None:
        row = {
            "network": self.verifier_string,
            "image_id": image_id,
            "true_label": true_label,
            "cert_status": cert_status,
            "cert_pred": cert_pred,
            "cert_radius_l2": cert_radius_l2,
            "adv_status": adv_status,
            "adv_pred": adv_pred,
            "within_cert_ball": int(within_cert_ball),
            "success_within_cert": int(success_within_cert),
            "min_adv_radius_l2": float(min_adv_radius_l2) if min_adv_radius_l2 is not None else np.nan,
            "attack": attack_name,
            "attack_eps_l2": attack_eps_l2,
            "step_size": step_size,
            "num_iter": num_iter,
            "eot_samples": eot_samples,
            "linf": linf,
            "l2": l2,
            "total_time": total_time,
            "artifact_path": artifact_path if artifact_path is not None else np.nan,
        }

        df = pd.DataFrame([row])
        df.to_csv(self.all_results_df_path, mode="a", header=False, index=False)

        if success_within_cert:
            df.to_csv(self.results_df_path, mode="a", header=False, index=False)

        self._success_flags.append(bool(success_within_cert))
        self._linf_values.append(float(linf))
        self._l2_values.append(float(l2))

    def create_summary(
        self,
        total_num: int,
        n_certified: int,
        n_attacked: int,
        n_success: int,
        avg_cert_radius_l2: float,
        step_size: float,
        num_iter: int,
        eot_samples: int,
        model_name: str,
        total_duration: float = 0.0,
    ) -> None:
        success_rate = (n_success / float(n_attacked)) if n_attacked > 0 else 0.0
        avg_linf = float(np.mean(self._linf_values)) if self._linf_values else 0.0
        avg_l2 = float(np.mean(self._l2_values)) if self._l2_values else 0.0

        summary_data = {
            "total_num": [total_num],
            "n_certified": [n_certified],
            "n_attacked": [n_attacked],
            "n_success": [n_success],
            "success_rate": [success_rate],
            "avg_cert_radius_l2": [avg_cert_radius_l2],
            "step_size": [step_size],
            "num_iter": [num_iter],
            "eot_samples": [eot_samples],
            "avg_linf": [avg_linf],
            "avg_l2": [avg_l2],
            "model_name": [model_name],
            "total_duration_seconds": [total_duration],
        }
        pd.DataFrame(summary_data).to_csv(self.summary_df_path, index=False)
        print(f"Saved attack summary dataframe to {self.summary_df_path}")

    def log_to_comet(self, tracker) -> None:
        tracker.log_asset(str(self.results_df_path))
        tracker.log_asset(str(self.all_results_df_path))
        tracker.log_asset(str(self.summary_df_path))


