import os, sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
import torch
from tqdm import tqdm
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datasets import load_dataset
from diffusers.models.activations import GEGLU
from diffusers import StableDiffusionPipeline
from transformers.models.clip.modeling_clip import CLIPMLP
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader, load_uncond_backdoored_model, get_villan_dataset, load_clean_villan_pipe
from utils.utils import *
from wanda_receiver import Wanda
from evaluation.configs.bdmodel_path import get_bdmodel_dict, set_bd_config
from utils.prompts import get_promptsPairs_fromDataset_bdInfo, get_bdPrompts_fromVillanDataset_random
from backdoor_loss import compute_neuron_sensitivity, compute_sd_neuron_sensitivity, approx_sd_neuron_sensitivity
import seaborn as sns
import json

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def process_layer_names(layer_names):
    new = []
    for i in layer_names:
        i = i[:-2]
        new.append(i)
    return new


def plot_neuron_distribution(k, preact, bd_preact, save_path, layer_name, target_t, bins=30, alpha=0.6, bd=True):
    if isinstance(preact, torch.Tensor):
        preact = preact.cpu().numpy()
    if isinstance(bd_preact, torch.Tensor):
        bd_preact = bd_preact.cpu().numpy()
    
    clean_values = preact[:, k]
    poisoned_values = bd_preact[:, k]
    if bd:
        neuron_type = 'Backdoor'
    else:
        neuron_type = 'Clean'
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(clean_values, bins=bins, alpha=alpha, label='Clean Samples', color='blue', density=True)
    plt.hist(poisoned_values, bins=bins, alpha=alpha, label='Poisoned Samples', color='red', density=True)
    
    sns.kdeplot(clean_values, color='blue', label='Clean Samples (KDE)', linewidth=2)
    sns.kdeplot(poisoned_values, color='red', label='Poisoned Samples (KDE)', linewidth=2)
    
    plt.xlabel(f'Pre-activation Value of Neuron {k}', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Pre-activation Distribution of {neuron_type} Neuron {k}', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = f'{save_path}/{target_t}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = f'{save_path}/layer{layer_name}_neuron{k}_preactivation.svg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
    

def main():
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--multi_target', type=bool, default=False)
    parser.add_argument('--base_config', type=str, default='./evaluation/configs/eval_config.yaml')
    parser.add_argument('--backdoor_method', type=str, default='badt2i_object')
    parser.add_argument('--result_dir', type=str, default='/mnt/sdb/linweilin/BackdoorDM/results/badt2i_object_sd15')
    parser.add_argument('--backdoored_model_path', type=str) # baddiffusion_DDPM-CIFAR10-32_0
    parser.add_argument('--extra_config', type=str, default=None) # extra config for some sampling methods
    
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--bd_config', type=str)
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--hook_module', type=str, default='preactivation')
    parser.add_argument('--timesteps', type=int, default=51) # infer steps 50/51 for t2i
    parser.add_argument('--keep_nsfw', type=bool, default=False)
    parser.add_argument('--sample_n', type=int, default=100)
    parser.add_argument('--k_layers', type=int, default=1)
    parser.add_argument('--n_neurons', type=int, default=5)
    parser.add_argument('--target_t', type=int, default=10) # 999
    
    cmd_args = parser.parse_args()
    if cmd_args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion']:
        args = base_args_uncond_v2(cmd_args)
        logger = set_logging(f'{args.backdoored_model_path}/sample_logs/')
        dsl = get_uncond_data_loader(config=args, logger=logger)
        set_random_seeds(args.seed)
        model, sched = load_uncond_backdoored_model(args)
        config_path = f"{args.backdoored_model_path}/config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        ckpt_path = config.get("ckpt")
        args.ckpt = ckpt_path
        model_clean, sched_clean = load_uncond_backdoored_model(args)
        args.ckpt = args.backdoored_model_path
        layer_names, top_n, least_n = compute_neuron_sensitivity(args, dsl, model, model_clean, sched, args.sample_n, args.k_layers, args.n_neurons)
        print(top_n)
        print(least_n)
        exit()
        
        args.replace_fn = torch.nn.modules.conv.Conv2d

        # Make two separate norm calculator classes for base and adj prompts
        neuron_receiver_base = Wanda(args.seed, args.timesteps, args.k_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module, target_t=args.target_t)
        neuron_receiver_target = Wanda(args.seed, args.timesteps, args.k_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module, target_t=args.target_t)
        result_dir = f'{args.result_dir}/analysis/pre_activations/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)

        sample_n = args.sample_n
        init = torch.randn(
                (sample_n, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size),
                generator=torch.Generator(args.device),
                device=args.device
            )
        bd_init = init + dsl.trigger.unsqueeze(0).to(args.device)
        neuron_receiver_base.reset_time_layer()
        out = neuron_receiver_base.observe_uncond_activation(args, model, init, dsl.image_size, sample_n, sched, False, target_module=layer_names)
        neuron_receiver_target.reset_time_layer()
        if args.backdoor_method == 'trojdiff':
            out_target = neuron_receiver_target.observe_uncond_activation(args, model, init, dsl.image_size, sample_n, sched, True, target_module=layer_names)
        else:
            out_target = neuron_receiver_target.observe_uncond_activation(args, model, bd_init, dsl.image_size, sample_n, sched, False, target_module=layer_names)
        # save
        act_norms_base = neuron_receiver_base.activation_norm.get_column_norms()
        act_norms_target = neuron_receiver_target.activation_norm.get_column_norms()
        neuron_receiver_base.activation_norm.save(os.path.join(result_dir,'base_norms.pt'))
        neuron_receiver_target.activation_norm.save(os.path.join(result_dir,'target_norms.pt'))
        
        cnt = 0
        for key, val in top_n.items():
            val_plot = val[:args.n_neurons]
            preact = act_norms_base[args.target_t][cnt]
            bd_preact = act_norms_target[args.target_t][cnt]
            for i in range(len(val_plot)):
                plot_neuron_distribution(val[i], preact, bd_preact, result_dir, key, target_t=str(args.target_t), bd=True)
            cnt += 1

        cnt = 0
        for key, val in least_n.items():
            val_plot = val[:args.n_neurons]
            preact = act_norms_base[args.target_t][cnt]
            bd_preact = act_norms_target[args.target_t][cnt]
            for i in range(len(val_plot)):
                plot_neuron_distribution(val[i], preact, bd_preact, result_dir, key, target_t=str(args.target_t), bd=False)
            cnt += 1

        # extract bd neurons and clean neurons
        
    else:
        if cmd_args.backdoor_method == 'villandiffusion_cond':
            cmd_args.bd_config = './attack/t2i_gen/configs/bd_config_fix.yaml'
            cmd_args.backdoored_model_path = cmd_args.result_dir
            args = base_args(cmd_args)
            args.val_data = 'CELEBA_HQ_DIALOG'
            args.caption_column = 'text'
            ds = get_villan_dataset(args)
            clean_prompts = ds[args.caption_column][:args.sample_n]
            bd_prompts = get_bdPrompts_fromVillanDataset_random(args, clean_prompts, args.sample_n)
            model = load_t2i_backdoored_model(args)
            model_clean = load_clean_villan_pipe(args.sched, args.clean_model_path)
        else:
            if cmd_args.bd_config is None:
                set_bd_config(cmd_args)
            args = base_args_v2(cmd_args)
            if getattr(args, 'backdoored_model_path', None) is None:
                args.backdoored_model_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method])
            ds = load_dataset(args.val_data)['train']
            ds_txt = ds[args.caption_colunm]
            args.test_robust_type = None
            bd_prompts, clean_prompts, _ = get_promptsPairs_fromDataset_bdInfo(args, ds_txt, args.sample_n)
            bd_prompts = bd_prompts[0]
            clean_prompts = clean_prompts[0]
            model = load_t2i_backdoored_model(args)
            model_clean = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None)
        
        set_random_seeds(args.seed)
        # replace_fn = GEGLU
        replace_fn = torch.nn.modules.conv.Conv2d
        args.replace_fn = replace_fn
        result_dir = f'{args.result_dir}/analysis/pre_activations/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        if os.path.exists(f'{result_dir}/top_n.json'):
            # layer_names = ['down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2']
            layer_names = ['down_blocks.0.resnets.0.conv1']
            with open(f'{result_dir}/top_n.json', 'r') as f:
                top_n = json.load(f)
            with open(f'{result_dir}/least_n.json', 'r') as f:
                least_n = json.load(f)
        else:
            layer_names, top_n, least_n = approx_sd_neuron_sensitivity(args, model, model_clean, bd_prompts, clean_prompts, args.sample_n, args.k_layers, args.n_neurons)
        print(top_n)
        print(least_n)
        with open(f'{result_dir}/top_n.json', 'w') as f:
            json.dump(top_n, f)
        with open(f'{result_dir}/least_n.json', 'w') as f:
            json.dump(least_n, f)
        # top_n = {'down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2': [16, 55, 50, 44, 24]}
        # least_n = {'down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2': [4, 17, 30, 23, 29]}
        
        print("Layer names: ", layer_names, len(layer_names))
        num_layers = len(layer_names)
        # layer_names = process_layer_names(layer_names)
        # Make two separate norm calculator classes for base and adj prompts
        neuron_receiver_base = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module, target_t=args.target_t)
        neuron_receiver_target = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module, target_t=args.target_t)
       
        
        for ann, ann_target in tqdm(zip(clean_prompts, bd_prompts)):
            print("text:", ann, ann_target)
            neuron_receiver_base.reset_time_layer()
            out = neuron_receiver_base.observe_activation(model, ann, layer_names)
            neuron_receiver_target.reset_time_layer()
            out_target = neuron_receiver_target.observe_activation(model, ann_target, layer_names)
        
        act_norms_base = neuron_receiver_base.activation_norm.get_column_norms()
        act_norms_target = neuron_receiver_target.activation_norm.get_column_norms()
        # save
        neuron_receiver_base.activation_norm.save(os.path.join(result_dir,'base_norms.pt'))
        neuron_receiver_target.activation_norm.save(os.path.join(result_dir,'target_norms.pt'))
         
        torch.save(act_norms_base, os.path.join(result_dir,'base_norms.pt')) # activation norm
        torch.save(act_norms_target, os.path.join(result_dir,'target_norms.pt'))

        cnt = 0
        for key, val in top_n.items():
            val_plot = val[:args.n_neurons]
            preact = act_norms_base[args.target_t][cnt]
            bd_preact = act_norms_target[args.target_t][cnt]
            for i in range(len(val_plot)):
                plot_neuron_distribution(val[i], preact, bd_preact, result_dir, key, target_t=str(args.target_t), bd=True)
            cnt += 1

        cnt = 0
        for key, val in least_n.items():
            val_plot = val[:args.n_neurons]
            preact = act_norms_base[args.target_t][cnt]
            bd_preact = act_norms_target[args.target_t][cnt]
            for i in range(len(val_plot)):
                plot_neuron_distribution(val[i], preact, bd_preact, result_dir, key, target_t=str(args.target_t), bd=False)
            cnt += 1

if __name__ == '__main__':
    main()

