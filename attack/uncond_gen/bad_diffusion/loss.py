# %%
from functools import partial
from sched import scheduler

import torch
from torch import nn
import torch.nn.functional as F
import os, sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())

from attack.uncond_gen.baddiff_backdoor import BadDiff_Backdoor
from utils.uncond_dataset import DEFAULT_VMAX, DEFAULT_VMIN
    
def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x_start)
        
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))

    alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
    alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = timesteps.to(x_start.device)

    sqrt_alphas_cumprod_t = alphas_cumprod[timesteps] ** 0.5                      # 根号下alpha bar t
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[timesteps]) ** 0.5      # 根号下1-alpha bar t 
    R_coef_t = (1 - alphas[timesteps] ** 0.5) * sqrt_one_minus_alphas_cumprod_t / (1 - alphas[timesteps])
    
    sqrt_alphas_cumprod_t = unqueeze_n(sqrt_alphas_cumprod_t)
    R_coef_t = unqueeze_n(R_coef_t)
    
    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)

    # return sqrt_alphas_cumprod_t * x_start + (1 - sqrt_alphas_cumprod_t) * R + sqrt_one_minus_alphas_cumprod_t * noise, R_coef_t * R + noise 
    # print(f"x_start shape: {x_start.shape}")
    # print(f"R shape: {R.shape}")
    # print(f"timesteps shape: {timesteps.shape}")
    # print(f"noise shape: {noise.shape}")
    # print(f"noisy_images shape: {noisy_images.shape}")
    # print(f"sqrt_alphas_cumprod_t shape: {sqrt_alphas_cumprod_t.shape}")
    # print(f"R_coef_t shape: {R_coef_t.shape}")
    return noisy_images + (1 - sqrt_alphas_cumprod_t) * R, R_coef_t * R + noise 

def p_losses_diffuser(noise_sched, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, loss_type: str="l2") -> torch.Tensor:
    if len(x_start) == 0: 
        return 0
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy, target = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise)
    # if clip:
    #     x_noisy = torch.clamp(x_noisy, min=DEFAULT_VMIN, max=DEFAULT_VMAX)
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
# %%
if __name__ == '__main__':
    # You can use the following code to visualize the forward process
    import os, sys
    
    from diffusers import DDPMScheduler

    sys.path.append('../')
    sys.path.append('../../')
    sys.path.append('../../../')
    sys.path.append(os.getcwd())
    from utils.uncond_model import DiffuserModelSched
    from utils.uncond_dataset import DatasetLoader
    
    time_step = 95
    num_train_timesteps = 100
    # time_step = 140
    # num_train_timesteps = 150
    ds_root = os.path.join('datasets')
    dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CELEBA_HQ).set_poison(trigger_type=BadDiff_Backdoor.TRIGGER_GLASSES, target_type=BadDiff_Backdoor.TARGET_CAT, clean_rate=1, poison_rate=0.2).prepare_dataset()
    print(f"Full Dataset Len: {len(dsl)}")
    image_size = dsl.image_size
    channels = dsl.channel
    ds = dsl.get_dataset()
    # CIFAR10
    # sample = ds[50000]
    # MNIST
    # sample = ds[60000]
    # CelebA-HQ
    # sample = ds[10000]
    sample = ds[24000]
    
    target = torch.unsqueeze(sample[DatasetLoader.TARGET], dim=0)
    source = torch.unsqueeze(sample[DatasetLoader.PIXEL_VALUES], dim=0)
    bs = len(source)
    model, noise_sched = DiffuserModelSched.get_model_sched(image_size=image_size, channels=channels, model_type=DiffuserModelSched.MODEL_DEFAULT)
    
    print(f"bs: {bs}")
    
    # Sample a random timestep for each image
    # mx_timestep = noise_sched.num_train_timesteps
    # timesteps = torch.randint(0, mx_timestep, (bs,), device=source.device).long()
    timesteps = torch.tensor([time_step] * bs, device=source.device).long()
    
    
    print(f"target Shape: {target.shape}")
    dsl.show_sample(img=target[0])
    print(f"source Shape: {source.shape}")
    dsl.show_sample(img=source[0])
    
    noise = torch.randn_like(target)
    noise_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    noisy_images = noise_sched.add_noise(source, noise, timesteps)
    
    noisy_x, target_x = q_sample_diffuser(noise_sched, x_start=target, R=source, timesteps=timesteps, noise=noise)
    
    print(f"target_x Shape: {target_x.shape}")
    dsl.show_sample(img=target_x[0], vmin=torch.min(target_x), vmax=torch.max(target_x))
    print(f"noisy_x Shape: {noisy_x.shape}")
    dsl.show_sample(img=noisy_x[0], vmin=torch.min(noisy_x), vmax=torch.max(noisy_x))
    print(f"source Shape: {source.shape}")
    dsl.show_sample(img=source[0], vmin=torch.min(source), vmax=torch.max(source))
    diff = (noisy_x - source)
    print(f"noisy_x - source Shape: {diff.shape}")
    dsl.show_sample(img=diff[0], vmin=torch.min(diff), vmax=torch.max(diff))
    diff = (target_x - noise)
    print(f"target_x - noise Shape: {diff.shape}")
    dsl.show_sample(img=diff[0], vmin=torch.min(diff), vmax=torch.max(diff))
    
    print(f"noisy_images Shape: {noisy_images.shape}")
    dsl.show_sample(img=noisy_images[0], vmin=torch.min(noisy_images), vmax=torch.max(noisy_images))
    diff_x = noisy_x - noisy_images
    print(f"noisy_x - noisy_images Shape: {diff_x.shape}")
    dsl.show_sample(img=diff_x[0], vmin=torch.min(diff_x), vmax=torch.max(diff_x))
    
    diff_x = noisy_x - target_x
    print(f"noisy_x - target_x Shape: {diff_x.shape}")
    dsl.show_sample(img=diff_x[0], vmin=torch.min(diff_x), vmax=torch.max(diff_x))# %%

# # %%
