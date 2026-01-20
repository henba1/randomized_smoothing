"""Hydra adversarial attack runner (PGD-EOT) for randomized smoothing defenses."""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import torch

from ada_verona import (
    BinarySearchEpsilonValueEstimator,
    create_experiment_directory,
    EOTPGDAttack,
    DataPoint,
    EpsilonStatus,
    VerificationContext,
    VerificationResult,
    One2AnyPropertyGenerator,
    get_balanced_sample,
    get_dataset_config,
    get_dataset_dir,
    get_models_dir,
    get_results_dir,
    get_sample,
    save_original_indices
)

from omegaconf import DictConfig, OmegaConf

from shared.utils.diffusion_timestep import find_t_for_sigma
from shared.core import Smooth
from shared.io.attack_result_writer import AttackCSVWriter
from shared.io.signal_handler import setup_signal_handler
from shared.tracking.comet_tracker import CometTracker
from shared.utils.attack_utils import FixedTModel, MinRadiusAttackVerifier, NamedNetwork, try_save_images
from shared.io.certify_run_loader import load_certify_run, make_dataset_subset_from_image_ids
from shared.utils.utils import get_device_with_diagnostics, get_diffusion_model_path_name_tuple


def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    start_time = time.time()
    device = get_device_with_diagnostics()

    dataset_name = cfg.dataset_name
    split = cfg.split
    sample_size = int(cfg.sample_size)
    random_seed = int(cfg.random_seed)
    sample_stratified = bool(cfg.get("sample_stratified", cfg.get("stratified", False)))

    sigma = float(cfg.sigma)
    classifier_type = str(cfg.classifier_type)
    classifier_name = str(cfg.classifier_name)
    classifier_name_short = classifier_name.split("/")[-1] if classifier_name else "unknown"
    pytorch_normalization = str(cfg.get("pytorch_normalization", "none"))

    experiment_type = str(cfg.get("experiment_type", "pgd_eot_attack"))
    experiment_tag = cfg.get("experiment_tag", None)

    attack_cfg = cfg.attack
    attack_only_if_correct = bool(attack_cfg.get("attack_only_if_correct", True))
    sigma_target_multiplier = float(attack_cfg.get("sigma_target_multiplier", 2.0))
    eval_n = int(attack_cfg.get("eval_n", 10000))
    eval_alpha = float(attack_cfg.get("eval_alpha", 0.001))
    eval_batch_size = int(attack_cfg.get("eval_batch_size", cfg.get("batch_size", 200)))
    within_cert_tol = float(attack_cfg.get("within_cert_tol", 1e-6))
    cert_N0 = int(attack_cfg.get("cert_N0", cfg.get("N0", 100)))
    cert_N = int(attack_cfg.get("cert_N", cfg.get("N", 100000)))
    cert_alpha = float(attack_cfg.get("cert_alpha", cfg.get("alpha", 0.001)))
    cert_batch_size = int(attack_cfg.get("cert_batch_size", cfg.get("batch_size", 200)))
    step_size_rel = attack_cfg.get("step_size_rel", None)
    step_size_abs = attack_cfg.get("step_size", None)
    bounds_cfg = attack_cfg.get("bounds", None)
    bounds = None if bounds_cfg in (None, "none") else (float(bounds_cfg[0]), float(bounds_cfg[1]))
    # Randomized smoothing certifies robustness in the input space of `x` passed to the pipeline (pixel space).
    # In this RS pipeline, any dataset normalization happens *inside* the classifier, so we must NOT rescale epsilons.
    random_start = bool(attack_cfg.get("random_start", False))
    norm = str(attack_cfg.get("norm", "l2"))
    attack_num_iter = int(attack_cfg.get("num_iter", 100))
    attack_eot_samples = int(attack_cfg.get("eot_samples", 20))

    if classifier_type == "onnx":
        raise ValueError("PGD-EOT requires gradients; ONNX classifier is not differentiable in this setup.")

    dataset_config_map = get_dataset_config()
    if dataset_name not in dataset_config_map:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'. Supported: {', '.join(dataset_config_map.keys())}")
    dataset_config = dataset_config_map[dataset_name]
    image_size = dataset_config["default_size"]
    num_classes = int(dataset_config["num_classes"])

    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name)
    RESULTS_DIR = get_results_dir(dataset_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = f"{classifier_name_short}_{dataset_name}_{experiment_type}_{timestamp}"
    _, ddpm_model_name = get_diffusion_model_path_name_tuple(dataset_name)

    tracker = CometTracker(
        experiment_name,
        dataset_name,
        classifier_name_short,
        ddpm_model_name,
        experiment_type=experiment_type,
        sigma=sigma,
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

    results_df_path = experiment_folder / "successful_attacks.csv"
    all_results_df_path = experiment_folder / "all_attacks.csv"
    summary_df_path = experiment_folder / "summary.csv"
    output_file = experiment_folder / f"{experiment_name}.txt"
    images_dir = experiment_folder / "images"

    verifier_string = f"RS_{classifier_name_short}_{ddpm_model_name}_sigma_{sigma}"
    csv_writer = AttackCSVWriter(
        results_df_path=results_df_path,
        all_results_df_path=all_results_df_path,
        summary_df_path=summary_df_path,
        verifier_string=verifier_string,
    )

    # Optional: search for the minimum L2 radius where an adversarial exists (may be outside certified radius).
    search_cfg = attack_cfg.get("min_radius_search", {})
    search_eps_start = search_cfg.get("epsilon_start", None)
    search_eps_stop = search_cfg.get("epsilon_stop", None)
    search_eps_step = search_cfg.get("epsilon_step", None)
    search_eval_n = int(search_cfg.get("eval_n", max(200, min(eval_n, 2000))))
    search_eval_alpha = float(search_cfg.get("eval_alpha", eval_alpha))
    search_eval_batch_size = int(search_cfg.get("eval_batch_size", eval_batch_size))
    search_num_iter = int(search_cfg.get("num_iter", min(attack_num_iter, 40)))
    search_eot_samples = int(search_cfg.get("eot_samples", min(attack_eot_samples, 10)))
    search_restarts = int(search_cfg.get("restarts", 2))

    certify_run_cfg = attack_cfg.get("certify_run", {})
    use_certify_run = bool(certify_run_cfg.get("enabled", False))
    certify_run_path = certify_run_cfg.get("path", None)
    certify_run_sigma_tol = float(certify_run_cfg.get("sigma_tol", 1e-6))

    tracker.log_parameters(
        {
            "dataset_name": dataset_name,
            "split": split,
            "sample_size": sample_size,
            "random_seed": random_seed,
            "sample_stratified": sample_stratified,
            "sigma": sigma,
            "sigma_target_multiplier": sigma_target_multiplier,
            "classifier_type": classifier_type,
            "classifier_name": classifier_name,
            "attack": "pgd_eot",
            "attack_only_if_correct": attack_only_if_correct,
            "cert_N0": cert_N0,
            "cert_N": cert_N,
            "cert_alpha": cert_alpha,
            "cert_batch_size": cert_batch_size,
            "eval_n": eval_n,
            "eval_alpha": eval_alpha,
            "eval_batch_size": eval_batch_size,
            "step_size_rel": step_size_rel,
            "step_size_abs": step_size_abs,
            "bounds": bounds,
            "std_rescale_factor": None,
            "attack_num_iter": attack_num_iter,
            "attack_eot_samples": attack_eot_samples,
            "within_cert_tol": within_cert_tol,
            "min_radius_search_epsilon_start": search_eps_start,
            "min_radius_search_epsilon_stop": search_eps_stop,
            "min_radius_search_epsilon_step": search_eps_step,
            "min_radius_search_eval_n": search_eval_n,
            "min_radius_search_eval_alpha": search_eval_alpha,
            "min_radius_search_eval_batch_size": search_eval_batch_size,
            "min_radius_search_num_iter": search_num_iter,
            "min_radius_search_eot_samples": search_eot_samples,
            "min_radius_search_restarts": search_restarts,
            "certify_run_enabled": use_certify_run,
            "certify_run_path": str(certify_run_path) if certify_run_path is not None else None,
            "certify_run_sigma_tol": certify_run_sigma_tol,
        }
    )

    if dataset_name == "CIFAR-10":
        from ds_cifar10.DRM import DiffusionRobustModel
    elif dataset_name == "ImageNet":
        from ds_imagenet.DRM import DiffusionRobustModel
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = DiffusionRobustModel(
        classifier_type=classifier_type,
        classifier_name=classifier_name,
        models_dir=MODELS_DIR,
        dataset_name=dataset_name,
        device=device,
        image_size=image_size,
        pytorch_normalization=pytorch_normalization,
    )

    t = find_t_for_sigma(diffusion=model.diffusion, sigma=sigma, target_multiplier=sigma_target_multiplier)
    attack_model = FixedTModel(model, t)
    smoothed = Smooth(
        model,
        num_classes,
        sigma,
        t,
        sample_correct_predictions=bool(cfg.get("sample_correct_predictions", True)),
    )
    # Define property generator and network wrapper for the attack
    property_generator = One2AnyPropertyGenerator()
    dummy_network = NamedNetwork(name=verifier_string)

    certify_samples = None
    sample_size_effective = int(sample_size)
    if use_certify_run:
        if certify_run_path is None:
            raise ValueError("attack.certify_run.enabled=True requires attack.certify_run.path to be set.")

        cert_data = load_certify_run(Path(certify_run_path))
        certify_samples_full = cert_data.samples
        if sample_size_effective <= 0:
            sample_size_effective = len(certify_samples_full)
        else:
            sample_size_effective = min(sample_size_effective, len(certify_samples_full))
        certify_samples = certify_samples_full[:sample_size_effective]

        tracker.log_parameters(
            {
                "sample_size_requested": int(sample_size),
                "sample_size_effective": int(sample_size_effective),
                "certify_run_num_available": int(len(certify_samples_full)),
            }
        )

        if cert_data.summary_sigma is not None:
            sigma_diff = abs(float(cert_data.summary_sigma) - float(sigma))
            tracker.log_parameters(
                {
                    "certify_run_sigma": float(cert_data.summary_sigma),
                    "certify_run_sigma_diff": float(sigma_diff),
                    "certify_run_sigma_matches": int(sigma_diff <= certify_run_sigma_tol),
                }
            )
            if sigma_diff > certify_run_sigma_tol:
                print(
                    f"WARNING: certify_run sigma={cert_data.summary_sigma} does not match attack sigma={sigma} "
                    f"(diff={sigma_diff}, tol={certify_run_sigma_tol})."
                )

        dataset, original_indices = make_dataset_subset_from_image_ids(
            dataset_name=dataset_name,
            dataset_dir=DATASET_DIR,
            train_bool=(split == "train"),
            image_ids=[s.image_id for s in certify_samples],
            image_size=None,
            flatten=False,
        )
    else:
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
        sample_size=sample_size_effective if use_certify_run else sample_size,
        split=split,
    )
    tracker.log_asset(str(indices_file))

    total_num = 0
    n_certified = 0
    n_attacked = 0
    n_success = 0
    sum_cert_radius = 0.0

    def get_summary_params() -> dict:
        return {
            "total_num": total_num,
            "n_certified": n_certified,
            "n_attacked": n_attacked,
            "n_success": n_success,
            "avg_cert_radius_l2": (sum_cert_radius / float(n_certified)) if n_certified > 0 else 0.0,
            "step_size": step_size_abs,
            "num_iter": attack_num_iter,
            "eot_samples": attack_eot_samples,
            "model_name": classifier_name_short,
            "total_duration": time.time() - start_time,
        }

    setup_signal_handler(csv_writer, tracker, output_file, get_summary_params)

    with open(output_file, "w") as f:
        f.write("original_idx\tlabel\tcert_pred\tcert_radius_l2\tadv_pred\tsuccess_within_cert\tmax_abs_delta\tl2_delta\ttime\n")
        for i in range(len(dataset)):
            original_idx = int(original_indices[i])
            x, label = dataset[i]
            x = x.to(device)
            x_b = x.unsqueeze(0)

            before = time.time()
            if use_certify_run:
                assert certify_samples is not None
                cert_row = certify_samples[i]
                if cert_row.image_id != original_idx:
                    raise RuntimeError(
                        f"Internal mismatch: certify_samples[{i}].image_id={cert_row.image_id} "
                        f"but original_indices[{i}]={original_idx}."
                    )
                cert_pred_int = int(cert_row.predicted_class)
                cert_radius_l2 = float(cert_row.epsilon_value)
                cert_status = cert_pred_int
                if int(label) != int(cert_row.original_label):
                    print(
                        f"WARNING: dataset label mismatch for image_id={original_idx}: "
                        f"dataset_label={int(label)} vs certify_original_label={int(cert_row.original_label)}"
                    )
            else:
                cert_pred, cert_radius = smoothed.certify(
                    x,
                    n0=cert_N0,
                    n=cert_N,
                    alpha=cert_alpha,
                    batch_size=cert_batch_size,
                    label=int(label) if attack_only_if_correct else None,
                )
                cert_pred_int = int(cert_pred)
                cert_radius_l2 = float(cert_radius)
                cert_status = cert_pred_int

            should_attack = True
            if cert_pred_int in (Smooth.ABSTAIN, Smooth.MISCLASSIFIED):
                should_attack = False

            adv_pred_int = cert_pred_int
            adv_status = Smooth.ABSTAIN
            success_within_cert = False
            max_abs_delta = 0.0
            l2_delta = 0.0
            image_path: str | None = None
            attack_eps_l2 = 0.0
            attack_step_size = 0.0
            min_adv_radius_l2: float | None = None

            if should_attack:
                n_certified += 1
                sum_cert_radius += cert_radius_l2
                n_attacked += 1

                attack_eps_l2 = cert_radius_l2
                y_attack = torch.tensor([cert_pred_int], device=device, dtype=torch.long)
                attacker = EOTPGDAttack(
                    number_iterations=attack_num_iter,
                    eot_samples=attack_eot_samples,
                    rel_stepsize=float(step_size_rel) if step_size_rel is not None else None,
                    abs_stepsize=float(step_size_abs) if step_size_rel is None and step_size_abs is not None else None,
                    randomise=random_start,
                    norm=norm,
                    bounds=bounds,
                    std_rescale_factor=None, #not needed for RS
                )
                x_adv = attacker.execute(attack_model, x_b, y_attack, epsilon=float(attack_eps_l2))

                best_adv_x: torch.Tensor | None = None
                best_adv_pred_int: int | None = None
                best_adv_l2: float | None = None
                best_adv_linf: float | None = None

                effective_eps = float(attack_eps_l2)
                if step_size_rel is not None:
                    attack_step_size = float(step_size_rel) * effective_eps
                elif step_size_abs is not None:
                    attack_step_size = float(step_size_abs)

                adv_pred = smoothed.predict(x_adv.squeeze(0), n=eval_n, alpha=eval_alpha, batch_size=eval_batch_size)
                adv_pred_int = int(adv_pred)
                adv_status = adv_pred_int
                flipped = (adv_pred_int != Smooth.ABSTAIN) and (adv_pred_int != cert_pred_int)

                if flipped:
                    delta = x_adv - x_b
                    best_adv_x = x_adv
                    best_adv_pred_int = adv_pred_int
                    best_adv_linf = float(delta.abs().amax().detach().cpu())
                    best_adv_l2 = float(delta.view(delta.shape[0], -1).norm(p=2, dim=1).mean().detach().cpu())

                initial_success_within_cert = bool(
                    flipped and (best_adv_l2 is not None) and (best_adv_l2 <= (cert_radius_l2 + within_cert_tol))
                )

                # run min-radius search if we did not already find an adversarial within the certified region -
                # if we did find one, the RS certificate is already broken and we can skip binary search.
                do_search_this_sample = bool(not initial_success_within_cert)
                if do_search_this_sample:
                    verifier = MinRadiusAttackVerifier(
                        attack_model=attack_model,
                        x_b=x_b,
                        y_attack=y_attack,
                        cert_pred_int=cert_pred_int,
                        smoothed=smoothed,
                        num_iter=search_num_iter,
                        eot_samples=search_eot_samples,
                        step_size_rel=float(step_size_rel) if step_size_rel is not None else None,
                        step_size_abs=float(step_size_abs) if step_size_abs is not None else None,
                        bounds=bounds,
                        restarts=search_restarts,
                        eval_n=search_eval_n,
                        eval_alpha=search_eval_alpha,
                        eval_batch_size=search_eval_batch_size,
                        abstain_int=Smooth.ABSTAIN,
                    )

                    if search_eps_start is None or search_eps_stop is None or search_eps_step is None:
                        raise ValueError(
                            "min_radius_search requires an explicit epsilon schedule: "
                            "attack.min_radius_search.epsilon_start/epsilon_stop/epsilon_step"
                        )

                    start = float(search_eps_start)
                    stop = float(search_eps_stop)
                    step = float(search_eps_step)
                    if step <= 0:
                        raise ValueError("attack.min_radius_search.epsilon_step must be > 0")
                    if stop < start:
                        raise ValueError("attack.min_radius_search.epsilon_stop must be >= epsilon_start")

                    eps_list = []
                    eps = start
                    while eps < stop:
                        if eps > 0:
                            eps_list.append(float(eps))
                        eps += step
                    if eps_list:
                        estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=eps_list, verifier=verifier)
                        epsilon_status_list = [EpsilonStatus(x, None, None, estimator.verifier.name) for x in eps_list]

                        min_radius_tmp_path = experiment_folder / "min_radius_search" / str(original_idx)
                        ctx = VerificationContext(
                            network=dummy_network,
                            data_point=DataPoint(id=str(original_idx), label=int(label), data=x.detach().cpu()),
                            tmp_path=min_radius_tmp_path,
                            property_generator=property_generator,
                            save_epsilon_results=True,
                        )
                        estimator.binary_search(ctx, epsilon_status_list)

                        sat_values = [x.value for x in epsilon_status_list if x.result == VerificationResult.SAT]
                        min_adv_radius_l2 = float(min(sat_values)) if sat_values else None

                        if verifier.best_sat_x_adv is not None and verifier.best_sat_epsilon is not None:
                            best_adv_x = verifier.best_sat_x_adv
                            best_adv_pred_int = int(verifier.best_sat_pred_int) if verifier.best_sat_pred_int is not None else None
                            attack_eps_l2 = float(verifier.best_sat_epsilon)
                            if step_size_rel is not None:
                                attack_step_size = float(step_size_rel) * float(attack_eps_l2)
                            elif step_size_abs is not None:
                                attack_step_size = float(step_size_abs)
                            delta = best_adv_x - x_b
                            best_adv_linf = float(delta.abs().amax().detach().cpu())
                            best_adv_l2 = float(delta.view(delta.shape[0], -1).norm(p=2, dim=1).mean().detach().cpu())

                search_success_within_cert = bool(
                    min_adv_radius_l2 is not None and min_adv_radius_l2 <= (cert_radius_l2 + within_cert_tol)
                )
                success_within_cert = bool(initial_success_within_cert or search_success_within_cert)

                if best_adv_x is not None and best_adv_l2 is not None:
                    if best_adv_pred_int is not None:
                        adv_pred_int = int(best_adv_pred_int)
                        adv_status = adv_pred_int

                    max_abs_delta = float(best_adv_linf) if best_adv_linf is not None else 0.0
                    l2_delta = float(best_adv_l2) if best_adv_l2 is not None else 0.0

                    saved = try_save_images(images_dir, image_id=original_idx, x=x_b, x_adv=best_adv_x)
                    if saved:
                        image_path = str(saved[-1])
                        for p in saved:
                            tracker.log_asset(str(p))

                n_success += int(success_within_cert)

            after = time.time()
            total_num += 1
            duration = after - before
            time_elapsed = str(datetime.timedelta(seconds=duration))

            csv_writer.append_result(
                image_id=original_idx,
                true_label=int(label),
                cert_status=cert_status,
                cert_pred=cert_pred_int,
                cert_radius_l2=cert_radius_l2,
                adv_status=adv_status,
                adv_pred=adv_pred_int,
                success_within_cert=success_within_cert,
                min_adv_radius_l2=min_adv_radius_l2,
                attack_name="pgd_eot",
                attack_eps_l2=attack_eps_l2,
                step_size=attack_step_size,
                num_iter=attack_num_iter,
                eot_samples=attack_eot_samples,
                search_num_iter=search_num_iter,
                search_eot_samples=search_eot_samples,
                search_restarts=search_restarts,
                max_abs_delta=max_abs_delta,
                l2_delta=l2_delta,
                total_time=time_elapsed,
                image_path=image_path,
            )

            tracker.log_metrics(
                {
                    "subset_index": i,
                    "original_idx": original_idx,
                    "true_label": int(label),
                    "cert_pred": cert_pred_int,
                    "cert_radius_l2": cert_radius_l2,
                    "adv_pred": adv_pred_int,
                    "success_within_cert": int(success_within_cert),
                    "max_abs_delta": max_abs_delta,
                    "l2_delta": l2_delta,
                    "n_certified": n_certified,
                    "n_attacked": n_attacked,
                    "n_success": n_success,
                    "success_rate": (n_success / float(n_attacked)) if n_attacked > 0 else 0.0,
                },
                step=total_num,
            )
            if min_adv_radius_l2 is not None:
                tracker.log_metrics({"min_adv_radius_l2": float(min_adv_radius_l2)}, step=total_num)

            f.write(
                f"{original_idx}\t{label}\t{cert_pred_int}\t{cert_radius_l2:.6f}\t{adv_pred_int}\t"
                f"{int(success_within_cert)}\t{max_abs_delta:.6f}\t{l2_delta:.6f}\t{time_elapsed}\n"
            )
            f.flush()

    total_duration = time.time() - start_time
    csv_writer.create_summary(
        total_num=total_num,
        n_certified=n_certified,
        n_attacked=n_attacked,
        n_success=n_success,
        avg_cert_radius_l2=(sum_cert_radius / float(n_certified)) if n_certified > 0 else 0.0,
        step_size=float(step_size_abs) if step_size_abs is not None else (float(step_size_rel) if step_size_rel is not None else 0.0),
        num_iter=attack_num_iter,
        eot_samples=attack_eot_samples,
        model_name=classifier_name_short,
        total_duration=total_duration,
    )
    csv_writer.log_to_comet(tracker)
    tracker.log_asset(str(output_file))
    if tracker.is_active:
        tracker.end()


