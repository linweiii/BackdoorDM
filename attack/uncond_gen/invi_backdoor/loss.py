import torch
import torch.nn.functional as F
import os, sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())


def q_sample_diffuser(noise_sched, x_start, R, timesteps, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))

    alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
    alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = timesteps.to(x_start.device)

    sqrt_alphas_cumprod_t = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[timesteps]) ** 0.5

    prev_timesteps = timesteps - 1
    alphas_cumprod_t_prev = []
    for p_times in prev_timesteps:
        if p_times >= 0:
            alphas_cumprod_t_prev.append(alphas_cumprod[p_times])
        else:
            alphas_cumprod_t_prev.append(1.0)
    alphas_cumprod_t_prev = torch.tensor(alphas_cumprod_t_prev).to(device=x_start.device, dtype=x_start.dtype)

    R_coef_t = ((alphas_cumprod_t_prev ** 0.5) - sqrt_alphas_cumprod_t) / (
                (alphas_cumprod_t_prev ** 0.5) * sqrt_one_minus_alphas_cumprod_t - sqrt_alphas_cumprod_t * (
                    (1 - alphas_cumprod_t_prev) ** 0.5))

    sqrt_alphas_cumprod_t = unqueeze_n(sqrt_alphas_cumprod_t)
    R_coef_t = unqueeze_n(R_coef_t)

    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)

    return noisy_images + (1 - sqrt_alphas_cumprod_t) * R, R_coef_t * R + noise


def p_losses_diffuser(noise_sched, model, x_start, R, timesteps, noise=None, loss_type="l2"):
    if len(x_start) == 0:
        return 0

    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy, target = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise)
    predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]

    if loss_type == 'l1':
        loss = F.l1_loss(target, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(target, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(target, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

