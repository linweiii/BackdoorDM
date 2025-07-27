import os, sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import gc
import copy
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.models.activations import GEGLU
# from utils.load import load_t2i_backdoored_model, get_uncond_data_loader, load_uncond_backdoored_model
# from attack.uncond_gen.bad_diffusion.loss import q_sample_diffuser, p_losses_diffuser


def compute_backdoor_loss(args, dsl, model, model_clean, noise_sched, rng, sample_n=100, max_batch_n=256):
    model.eval()
    model_clean.eval()
    if args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion']:
        init = torch.randn(
                (sample_n, model.in_channels, model.sample_size, model.sample_size),
                generator=rng,
                device=args.device
            )
        trigger = dsl.trigger.unsqueeze(0).to(args.device)
        bd_init = init + trigger
        # init.to(model.device)
        # bd_init.to(model.device)
        init = torch.split(init, max_batch_n)
        bd_init = torch.split(bd_init, max_batch_n)
        batch_sizes = list(map(lambda x: len(x), init))
        cnt = 0
        total_loss = 0.0
        for i, batch_sz in enumerate(batch_sizes):
            timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (batch_sz,), device=model.device).long()
            with torch.no_grad():
                pred_bd = model(bd_init[i].contiguous(), timesteps.contiguous(), return_dict=False)[0]
                pred = model_clean(init[i].contiguous(), timesteps.contiguous(), return_dict=False)[0]
                loss = F.mse_loss(pred_bd, pred)
                total_loss += loss.item() * batch_sz
                cnt += batch_sz

        return total_loss / cnt
    else:
        raise NotImplementedError()
    
def move_pipe_to_cpu(pipe):
    pipe.unet.cpu()
    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    torch.cuda.empty_cache()
    return pipe

def get_sd_noise(pipe, prompt, init, timesteps, device):
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    init_shape = init.shape
    text_shape = text_embeddings.shape
    time_shape = timesteps.shape
    noise_pred = pipe.unet(init, timesteps, text_embeddings).sample
    return noise_pred
    
def compute_sd_backdoor_loss(args, pipe, pipe_clean, bd_prompts, clean_prompts, generator, sample_n, require_grad=False):
    # generator = torch.Generator(args.device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.clean_model_path, subfolder="scheduler", low_cpu_mem_usage=False, )
    latent_shape = (1, pipe.unet.config.in_channels, pipe.unet.sample_size, pipe.unet.sample_size)
    total_loss = 0.0
    cnt = 0
    for prompt, prompt_bd in tqdm(zip(clean_prompts[:sample_n], bd_prompts[:sample_n])):
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=args.device)
        timesteps = timesteps.long()
        init = torch.randn(latent_shape, generator=generator, device=args.device)
        pipe = pipe.to(args.device)
        if require_grad:
            noise_pred_bd = get_sd_noise(pipe, prompt_bd, init, timesteps, args.device)
        else:
            with torch.no_grad():
                noise_pred_bd = get_sd_noise(pipe, prompt_bd, init, timesteps, args.device)
        pipe = move_pipe_to_cpu(pipe)

        pipe_clean = pipe_clean.to(args.device)
        with torch.no_grad():
            noise_pred = get_sd_noise(pipe_clean, prompt, init, timesteps, args.device)
        pipe_clean = move_pipe_to_cpu(pipe)
        loss = F.mse_loss(noise_pred_bd, noise_pred)
        total_loss += loss.item()
        cnt += 1

    return total_loss / cnt


    
def top_n_sensitive(sensitivity_dict, n):
    top_indices_dict = {}
    least_indices_dict = {}
    for layer_name, sensitivities in sensitivity_dict.items():
        top_n_indices = sorted(range(len(sensitivities)), key=lambda i: sensitivities[i], reverse=True)[:n]
        top_indices_dict[layer_name] = top_n_indices

        least_n_indices = sorted(range(len(sensitivities)), key=lambda i: sensitivities[i])[:n]
        least_indices_dict[layer_name] = least_n_indices

    return top_indices_dict, least_indices_dict

    
def compute_neuron_sensitivity(args, dsl, model, model_clean, noise_sched, sample_n, k_layers=1, n_neurons=10):
    unet = model.unet
    unet_clean = model_clean.unet
    rng = torch.Generator(args.device)
    baseline_loss = compute_backdoor_loss(args, dsl, unet, unet_clean, noise_sched, rng, sample_n)
    cnt = 0
    sensitivity = {}
    names = []
    for name, module in unet.named_modules(): # capture first k conv layers
        # print(name)
        if isinstance(module, torch.nn.modules.conv.Conv2d) and 'conv' in name: # isinstance(module, torch.nn.modules.conv.Conv2d) and 
            if name == 'conv_in' or name == 'conv_out':
                continue
            if cnt == k_layers:
                break
            print(type(module))
            print(name)
            names.append(name)
            weight = module.weight.detach()
            weight_shape = weight.shape # [128, 3, 3, 3]
            num_neurons = weight.shape[0]
            for i in tqdm(range(num_neurons)):
                unet_copy = copy.deepcopy(unet)
                target_module = dict(unet_copy.named_modules())[name]
                new_weight = target_module.weight.detach().clone()
                new_weight[i, :, :, :] = 0.0
                target_module.weight.data = new_weight
                loss_pruned = compute_backdoor_loss(args, dsl, unet_copy, unet_clean, noise_sched, rng, sample_n)
                
                sensitivity_i = baseline_loss - loss_pruned
                if name in sensitivity:
                    sensitivity[name].append(sensitivity_i)
                else:
                    sensitivity[name] = [sensitivity_i]
                del unet_copy
                torch.cuda.empty_cache()
                gc.collect()
            cnt += 1

    top_n, least_n = top_n_sensitive(sensitivity, n_neurons)

    return names, top_n, least_n

def compute_sd_neuron_sensitivity(args, pipe, pipe_clean, bd_prompts, clean_prompts, sample_n, k_layers=1, n_neurons=10, stride=20):
    rng = torch.Generator(args.device)
    baseline_loss = compute_sd_backdoor_loss(args, pipe, pipe_clean, bd_prompts, clean_prompts, rng, sample_n)
    cnt = 0
    sensitivity = {}
    names = []
    for name, module in pipe.unet.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d) and 'conv' in name:
        # if isinstance(module, replace_fn) and 'ff.net' in name:
            if name == 'conv_in' or name == 'conv_out':
                    continue
            if cnt == k_layers:
                break
            print(type(module))
            print(name)
            names.append(name)
            weight = module.weight.detach()
            weight_shape = weight.shape # [320, 1280]
            num_neurons = weight.shape[-1]
            for i in tqdm(range(0, num_neurons, stride)):
                pipe_copy = copy.deepcopy(pipe)
                target_module = dict(pipe_copy.unet.named_modules())[name]
                new_weight = target_module.weight.detach().clone()
                new_weight[:, i] = 0.0
                target_module.weight.data = new_weight
                loss_pruned = compute_sd_backdoor_loss(args, pipe_copy, pipe_clean, bd_prompts, clean_prompts, rng, sample_n)
                sensitivity_i = baseline_loss - loss_pruned
                if name in sensitivity:
                    sensitivity[name].append(sensitivity_i)
                else:
                    sensitivity[name] = [sensitivity_i]
                del pipe_copy
                torch.cuda.empty_cache()
                gc.collect()
            cnt += 1
    top_n, least_n = top_n_sensitive(sensitivity, n_neurons)

    return names, top_n, least_n

def approx_sd_neuron_sensitivity(args, pipe, pipe_clean, bd_prompts, clean_prompts, sample_n, k_layers=1, n_neurons=10):
    rng = torch.Generator(args.device)
    pipe.to(args.device)
    cnt = 0
    sensitivity = {}
    names = []
    noise_scheduler = DDPMScheduler.from_pretrained(args.clean_model_path, subfolder="scheduler", low_cpu_mem_usage=False, )
    latent_shape = (1, pipe.unet.config.in_channels, pipe.unet.sample_size, pipe.unet.sample_size)
    cnt = 0
    for name, module in pipe.unet.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d) and 'conv' in name:
            if name == 'conv_in' or name == 'conv_out':
                continue
            if cnt == k_layers:
                break
            print(type(module))
            print(name)
            names.append(name)
            weight = module.weight.detach()
            weight_shape = weight.shape # [320, 1280]
            num_neurons = weight.shape[-1]
            total_grad = torch.zeros_like(weight)
            
            for prompt, prompt_bd in tqdm(zip(clean_prompts[:sample_n], bd_prompts[:sample_n])):
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=args.device)
                timesteps = timesteps.long()
                init = torch.randn(latent_shape, generator=rng, device=args.device)

                # pipe = pipe.to(args.device)
                noise_pred_bd = get_sd_noise(pipe, prompt_bd, init, timesteps, args.device)
                # pipe = move_pipe_to_cpu(pipe)

                pipe_clean = pipe_clean.to(args.device)
                with torch.no_grad():
                    noise_pred = get_sd_noise(pipe_clean, prompt, init, timesteps, args.device)
                pipe_clean = move_pipe_to_cpu(pipe_clean)
                loss = F.mse_loss(noise_pred_bd, noise_pred)
                loss.backward()
                total_grad += module.weight.grad.detach()
                pipe.unet.zero_grad()

            weight = module.weight.detach()
            avg_grad = total_grad / sample_n
            sensitivity_matrix = torch.abs(weight * avg_grad)
            shape1 = sensitivity_matrix.shape
            # neurons_sensitivity = sensitivity_matrix.sum(dim=0)
            neurons_sensitivity = torch.sum(sensitivity_matrix, dim=(1, 2, 3))
            shape2 = neurons_sensitivity.shape
            sensitivity[name] = neurons_sensitivity.cpu().numpy()
            cnt += 1
    top_n, least_n = top_n_sensitive(sensitivity, n_neurons)

    return names, top_n, least_n
