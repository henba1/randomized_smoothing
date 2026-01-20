from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn as nn

from ada_verona import EOTPGDAttack
from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData, VerificationResult


def try_save_images(output_dir: Path, *, image_id: int, x: torch.Tensor, x_adv: torch.Tensor) -> list[Path]:
    """
    Save clean/adversarial examples as PNG if torchvision is available, else as .pt tensors.

    Returns:
        List of written paths (typically [clean_path, adv_path]).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        from torchvision.utils import save_image

        x_path = output_dir / f"{image_id}_clean.png"
        adv_path = output_dir / f"{image_id}_adv.png"
        save_image(x, x_path)
        save_image(x_adv, adv_path)
        paths.extend([x_path, adv_path])
    except Exception:
        x_path = output_dir / f"{image_id}_clean.pt"
        adv_path = output_dir / f"{image_id}_adv.pt"
        torch.save(x.detach().cpu(), x_path)
        torch.save(x_adv.detach().cpu(), adv_path)
        paths.extend([x_path, adv_path])
    return paths


class FixedTModel(nn.Module):
    """Wrap a diffusion-robust model into a standard `nn.Module(x)->logits` by  fixing timestep `t`."""

    def __init__(self, drm: nn.Module, t: int) -> None:
        super().__init__()
        self.drm = drm
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drm(x, self.t, enable_grad=True)


@dataclass(frozen=True)
class NamedNetwork:
    """Minimal `network` object for ada_verona's VerificationContext logging."""

    name: str


class MinRadiusAttackVerifier:
    """
    Adapter for ada_verona epsilon estimators.

    For a given epsilon, runs (EOT-)PGD a few times (restarts) and returns SAT if any run
    flips the *smoothed* prediction away from the certified class (excluding ABSTAIN).
    """

    name = "min_radius_attack_search"

    def __init__(
        self,
        *,
        attack_model: nn.Module,
        x_b: torch.Tensor,
        y_attack: torch.Tensor,
        cert_pred_int: int,
        smoothed,
        # attack params
        num_iter: int,
        eot_samples: int,
        step_size_rel: float | None,
        step_size_abs: float | None,
        bounds: tuple[float, float] | None,
        restarts: int,
        # evaluation params for Smooth.predict
        eval_n: int,
        eval_alpha: float,
        eval_batch_size: int,
        abstain_int: int,
    ) -> None:
        self.attack_model = attack_model
        self.x_b = x_b
        self.y_attack = y_attack
        self.cert_pred_int = int(cert_pred_int)
        self.smoothed = smoothed

        self.num_iter = int(num_iter)
        self.eot_samples = int(eot_samples)
        self.step_size_rel = step_size_rel
        self.step_size_abs = step_size_abs
        self.bounds = bounds
        self.restarts = int(restarts)

        self.eval_n = int(eval_n)
        self.eval_alpha = float(eval_alpha)
        self.eval_batch_size = int(eval_batch_size)
        self.abstain_int = int(abstain_int)

        # Best (smallest-epsilon) adversarial found so far during the verifier calls.
        self.best_sat_epsilon: float | None = None
        self.best_sat_x_adv: torch.Tensor | None = None
        self.best_sat_pred_int: int | None = None

    def verify(self, verification_context: VerificationContext, epsilon: float) -> CompleteVerificationData:  # noqa: ARG002
        start = time.time()
        is_sat = False
        pred_try_int = self.abstain_int
        x_sat: torch.Tensor | None = None

        for _ in range(max(1, self.restarts)):
            attacker_eps = EOTPGDAttack(
                number_iterations=self.num_iter,
                eot_samples=self.eot_samples,
                rel_stepsize=float(self.step_size_rel) if self.step_size_rel is not None else None,
                abs_stepsize=float(self.step_size_abs)
                if self.step_size_rel is None and self.step_size_abs is not None
                else None,
                randomise=True,
                bounds=self.bounds,
                std_rescale_factor=None,
            )

            x_try = attacker_eps.execute(self.attack_model, self.x_b, self.y_attack, epsilon=float(epsilon))
            pred_try = self.smoothed.predict(
                x_try.squeeze(0),
                n=self.eval_n,
                alpha=self.eval_alpha,
                batch_size=self.eval_batch_size,
            )
            pred_try_int = int(pred_try)
            if (pred_try_int != self.abstain_int) and (pred_try_int != self.cert_pred_int):
                is_sat = True
                x_sat = x_try
                break

        if is_sat:
            eps_f = float(epsilon)
            if self.best_sat_epsilon is None or eps_f < self.best_sat_epsilon:
                self.best_sat_epsilon = eps_f
                self.best_sat_x_adv = x_sat.detach() if x_sat is not None else None
                self.best_sat_pred_int = int(pred_try_int)

        duration = time.time() - start
        return CompleteVerificationData(
            result=VerificationResult.SAT if is_sat else VerificationResult.UNSAT,
            took=duration,
            obtained_labels=[str(pred_try_int)],
        )

