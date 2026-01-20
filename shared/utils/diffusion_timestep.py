from __future__ import annotations


def find_t_for_sigma(*, diffusion, sigma: float, target_multiplier: float = 2.0) -> int:
    """
    Map smoothing noise level `sigma` to a diffusion timestep `t`.

    Diffusion forward process (q_sample) has the form:
      x_t = a_t * x_0 + b_t * eps,  eps ~ N(0, I)
    where:
      a_t = diffusion.sqrt_alphas_cumprod[t]
      b_t = diffusion.sqrt_one_minus_alphas_cumprod[t]

    The effective noise std in the (scaled) input domain is b_t / a_t.

    We pick the smallest `t` such that:
      (b_t / a_t) >= target_multiplier * sigma

    - Inputs to diffusion denoiser are mapped from [0,1] to [-1,1] before denoising 
    (scale factor 2), which is why `target_multiplier` defaults to 2.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if target_multiplier <= 0:
        raise ValueError("target_multiplier must be > 0")

    target_sigma = sigma * target_multiplier
    real_sigma = 0.0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = diffusion.sqrt_alphas_cumprod[t]
        b = diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = (b / a).item() if hasattr(b / a, "item") else float(b / a)
    return t

