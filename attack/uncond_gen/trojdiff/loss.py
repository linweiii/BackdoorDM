import torch

def trojdiff_loss(args, noise_sched, model, x0, y, t, miu, target_label, gamma, cond_prob=1, keepdim=False):
    target_idx = torch.where(y == target_label)[0]
    chosen_mask = torch.bernoulli(torch.zeros_like(target_idx) + cond_prob)
    chosen_target_idx = target_idx[torch.where(chosen_mask == 1)[0]]
    
    batch, device = x0.shape[0], x0.device
    miu_ = torch.stack([miu.to(device)] * batch)
    
    alphas_cumprod = noise_sched.alphas_cumprod.to(device=device, dtype=x0.dtype)
    # alphas = noise_sched.alphas.to(device=device, dtype=x0.dtype)
    t = t.to(device)
    alphas_cumprod_used = alphas_cumprod[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    x = x0 * alphas_cumprod_used ** 0.5 + noise * (1.0 - alphas_cumprod_used) ** 0.5
    x_ = x0 * alphas_cumprod_used ** 0.5 + (noise * (1.0 - alphas_cumprod_used) ** 0.5) * gamma + miu_ * (1.0 - alphas_cumprod_used) ** 0.5
    
    if args.trigger_type == 'patch':
        tmp_x = x.clone()
        tmp_x[:, :, -args.patch_size:, -args.patch_size:] = x_[:, :, -args.patch_size:, -args.patch_size:]
        x_ = tmp_x
        
    x_add = x_[chosen_target_idx]
    t_add = t[chosen_target_idx]
    noise_add = noise[chosen_target_idx]
    x = torch.cat([x, x_add], dim=0)
    t = torch.cat([t, t_add], dim=0)
    noise = torch.cat([noise, noise_add], dim=0)
    
    output = model(x, t.float(), return_dict=False)[0]
    if keepdim:
        return (noise - output).square().sum(dim=(1, 2, 3))
    else:
        return (noise - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    
def trojdiff_loss_out(args, noise_sched, model, x0, y, t, miu, gamma, cond_prob=1, keepdim=False):
    target_idx = torch.where(y == 1000)[0]
    chosen_mask = torch.bernoulli(torch.zeros_like(target_idx) + cond_prob)
    chosen_target_idx = target_idx[torch.where(chosen_mask == 1)[0]]
    
    batch, device = x0.shape[0], x0.device
    miu_ = torch.stack([miu.to(device)] * batch)
    
    alphas_cumprod = noise_sched.alphas_cumprod.to(device=device, dtype=x0.dtype)
    # alphas = noise_sched.alphas.to(device=device, dtype=x0.dtype)
    t = t.to(device)
    alphas_cumprod_used = alphas_cumprod[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    
    x = x0 * alphas_cumprod_used ** 0.5 + noise * (1.0 - alphas_cumprod_used) ** 0.5
    x_ = x0 * alphas_cumprod_used ** 0.5 + (noise * (1.0 - alphas_cumprod_used) ** 0.5) * gamma + miu_ * (1.0 - alphas_cumprod_used) ** 0.5
    
    if args.trigger_type == 'patch':
        tmp_x = x.clone()
        tmp_x[:, :, -args.patch_size:, -args.patch_size:] = x_[:, :, -args.patch_size:, -args.patch_size:]
        x_ = tmp_x
    
    x_add = x_[chosen_target_idx]
    x[chosen_target_idx] = x_add
    output = model(x, t.float(), return_dict=False)[0]
    if keepdim:
        return (noise - output).square().sum(dim=(1, 2, 3))
    else:
        return (noise - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    