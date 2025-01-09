import torch
from torch import nn

    
def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, last=True):
    if noise is None:
        noise = torch.randn_like(x_start)
    timesteps = timesteps.to(x_start.device)
    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)
    if last:
        return noisy_images + R, noise
    else:
        return noisy_images + 0*R, noise


def p_losses_diffuser(noise_sched, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, last =True):
    if len(x_start) == 0: 
        return 0
    noise_1 = torch.randn_like(x_start)
    noise_2 = torch.randn_like(x_start)
    x_noisy_1, target_1 = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise_1, last =last)
    x_noisy_2, target_2 = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise_2, last =last)
    predicted_noise_1 = model(x_noisy_1, timesteps, return_dict=False)[0]
    predicted_noise_2 = model(x_noisy_2, timesteps, return_dict=False)[0]
    loss = 0.5*(target_1-predicted_noise_1-(target_2-predicted_noise_2)).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return loss

def backdoor_reconstruct_loss(model,
                          x0: torch.Tensor,
                          gamma: torch.Tensor,
                          t: torch.LongTensor,
                          e1: torch.Tensor,
                          e2: torch.Tensor,
                          b: torch.Tensor,
                          miu: torch.Tensor,
                          keepdim=False,
                          surrogate=True):
    batch, device = x0.shape[0], x0.device
    miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    if surrogate:
        x_ = x0 * a.sqrt() + e1 * (1.0 - a).sqrt() * gamma + miu_
        x_1 = x0 * a.sqrt() + e2 * (1.0 - a).sqrt() * gamma + miu_
    else:
        x_ = x0 * a.sqrt() + e1 * (1.0 - a).sqrt() * gamma
        x_1 = x0 * a.sqrt() + e2 * (1.0 - a).sqrt() * gamma
    x_add = x_
    x_add_1 = x_1
    t_add = t
    e_add = e1
    e_add_1 = e2
    x = x_add
    x_1 = x_add_1
    t = t_add
    e = e_add
    e_1 = e_add_1
    output = model(x, t.float(), return_dict=False)[0]
    output_1 = model(x_1, t.float(), return_dict=False)[0]
    if keepdim:
        return 0.5*(e - output-(e_1-output_1)).square().sum(dim=(1, 2, 3))
    else:
        return 0.5*(e - output-(e_1-output_1)).square().sum(dim=(1, 2, 3)).mean(dim=0)