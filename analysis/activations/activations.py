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
from datasets import load_dataset
from diffusers.models.activations import GEGLU
from transformers.models.clip.modeling_clip import CLIPMLP
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader, load_uncond_backdoored_model
from utils.utils import *
from wanda_receiver import Wanda

backdoored_model_path_dict = {
        # ImageFix Backdoor
        'villandiffusion_cond': 'villandiffusion_cond_trigger-latte-coffee_target-cat',
  
        # ImagePatch Backdoor
        'badt2i_pixel': 'badt2i_pixel_trigger-u200b_target-boya',

        # ObjectRep Backdoor
        'badt2i_object': 'badt2i_object_trigger-u200b_target-cat',
        'eviledit': 'eviledit_trigger-beautifuldog_target-cat.pt',
        'rickrolling_TPA': 'rickrolling_TPA_trigger-ȏ_target-cat',
        'paas_db': 'paas_db_trigger-[V]dog_target-cat',
        'paas_ti': 'paas_ti_trigger-[V]dog_target-cat',

        # StyleAdd Backdoor
        'rickrolling_TAA': 'rickrolling_TAA_trigger-ȏ_target-black_and_white_photo',
        'badt2i_style': 'badt2i_style_trigger-u200b_target-blackandwhitephoto',

        # ObjectAdd Backdoor
        'eviledit_objectAdd': 'eviledit_objectAdd_trigger-beautifuldog_target-dogandazebra.pt',
        'eviledit_numAdd': 'eviledit_numAdd_trigger-beautifuldog_target-twodogs.pt',
        'badt2i_objectAdd': 'badt2i_objectAdd_trigger-u200b_target-dogandazebra',
        'eviledit_add': 'eviledit_add_trigger-beautiful_target-dog.pt',
    }

def get_activation_range(dict1, dict2, target_layers, time_steps):
    
    max_value = float('-inf')
    min_value = float('inf')

    for time_step in range(time_steps):
        for layer in target_layers:
            layer1_activations = dict1.get(time_step, {}).get(layer, None)
            layer2_activations = dict2.get(time_step, {}).get(layer, None)
            if layer1_activations is not None:
                max_value = max(max_value, torch.max(layer1_activations).item())
                min_value = min(min_value, torch.min(layer1_activations).item())
            if layer2_activations is not None:
                max_value = max(max_value, torch.max(layer2_activations).item())
                min_value = min(min_value, torch.min(layer2_activations).item())

    return max_value, min_value

def get_activation_range_single(dict1, target_layers, time_steps):
    
    max_value = float('-inf')
    min_value = float('inf')

    for time_step in range(time_steps):
        for layer in target_layers:
            layer1_activations = dict1.get(time_step, {}).get(layer, None)
            if layer1_activations is not None:
                max_value = max(max_value, torch.max(layer1_activations).item())
                min_value = min(min_value, torch.min(layer1_activations).item())

    return max_value, min_value

def visualize_layerwise_activation_norms(activation_dict, time_step, selected_layers=None, bd=False, vmin=None, vmax=None, save_path=None, filename=None):
    time_step_data = activation_dict[time_step]
    
    if selected_layers is None:
        selected_layers = range(len(time_step_data))
    
    max_neurons = max(time_step_data[layer_index].shape[0] for layer_index in selected_layers)
    activations = []
    
    for layer_index in selected_layers:
        layer_data = time_step_data[layer_index].numpy()
        padded_data = np.pad(layer_data, (0, max_neurons - len(layer_data)), constant_values=np.nan)
        activations.append(padded_data)
    
    activations_matrix = np.vstack(activations).T
    block_width = 1
    plt.figure(figsize=(len(selected_layers) * block_width, 6))
    im = plt.imshow(activations_matrix, aspect='auto', cmap='viridis_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Activation Norm (L2)')
    
    plt.xlabel('Layer')
    plt.ylabel('Neuron')
    if filename == None:
        if bd:
            plt.title(f'Backdoored Model Activation Norms at Time Step {time_step}')
            filename = f'bd_activation_norms_{time_step}.svg'
        else:
            plt.title(f'Clean Model Activation Norms at Time Step {time_step}')
            filename = f'clean_activation_norms_{time_step}.svg'
    else:
        plt.title(f'Time Step {time_step}')
    plt.xticks(ticks=range(len(selected_layers)), labels=[f'{i}' for i in selected_layers])
    plt.gca().invert_yaxis()
    save_path = os.path.join(save_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
    

def visualize_timewise_activation_norms(activation_dict, layer_index, num_time_steps=10, bd=False, vmin=None, vmax=None, save_path=None, filename=None):
    all_time_steps = sorted(activation_dict.keys()) 
    time_steps = np.linspace(0, len(all_time_steps) - 1, num_time_steps, dtype=int)
    time_steps = [all_time_steps[i] for i in time_steps]
    activations = []
    max_neurons = 0
    for t in time_steps:
        layer_data = activation_dict[t][layer_index].numpy()
        max_neurons = max(max_neurons, len(layer_data))  # Track the max neuron count for padding
        activations.append(layer_data)
    
    activations_padded = [np.pad(act, (0, max_neurons - len(act)), constant_values=np.nan) for act in activations]
    activations_matrix = np.stack(activations_padded, axis=1)
    block_width = 1
    plt.figure(figsize=(len(time_steps) * block_width, 6))
    im = plt.imshow(activations_matrix, aspect='auto', cmap='viridis_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Activation Norm (L2)')
    
    plt.xlabel('Time Step')
    plt.ylabel('Neuron')
    if filename == None:
        if bd:
            plt.title(f'Backdoored Model Activation Norms for Layer {layer_index} Across Time Steps')
            filename = f'bd_activation_norms_layer{layer_index}.svg'
        else:
            plt.title(f'Clean Model Activation Norms for Layer {layer_index} Across Time Steps')
            filename = f'clean_activation_norms_layer{layer_index}.svg'
    else:
        plt.title(f'Clean vs Backdoored Activation Norms for Layer {layer_index}')
    plt.xticks(ticks=range(len(time_steps)), labels=[f'{t}' for t in time_steps])  
    plt.gca().invert_yaxis()
    save_path = os.path.join(save_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
    
def visualize_text_activation_norms(activation_dict, bd=False, vmin=None, vmax=None, save_path=None, filename=None):
    activations = []
    for i in activation_dict[0]:
        layer_data = activation_dict[0][i].numpy()
        activations.append(layer_data)
    
    activations_matrix = np.stack(activations, axis=1)
    plt.figure(figsize=(len(activation_dict[0]) * 1, 6))
    im = plt.imshow(activations_matrix, aspect='auto', cmap='viridis_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Activation Norm (L2)')
    
    plt.xlabel('Layer')
    plt.ylabel('Neuron')
    if filename == None:
        if bd:
            plt.title(f'Backdoored Text Encoder Activation Norms')
            filename = f'bd_txt_encoder_activation_norms.svg'
        else:
            plt.title(f'Clean Model Activation Norms')
            filename = f'clean_txt_encoder_activation_norms.svg'
    else:
        plt.title(f'Clean vs Backdoored Text Encoder Activation Norms')
    plt.xticks(ticks=range(len(activation_dict[0])), labels=[f'{i}' for i in activation_dict[0]])
    plt.gca().invert_yaxis()
    save_path = os.path.join(save_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
    

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='./evaluation/configs/eval_config.yaml')
    parser.add_argument('--backdoor_method', type=str, default='paas_db')
    parser.add_argument('--bd_target_type', type=str, default='objectRep')
    parser.add_argument('--result_dir', type=str, default='./results/paas_db_sd15')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--extra_config', type=str, default=None) # extra config for some sampling methods
    
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--bd_config', type=str, default='./attack/t2i_gen/configs/bd_config_objectRep.yaml')
    parser.add_argument('--clean_prompts', type=str, default='a dog sitting on the sofa') # a dog sitting on the sofa
    parser.add_argument('--bd_prompts', type=str, default='a [V] dog sitting on the sofa')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--hook_module', type=str, default='unet')
    parser.add_argument('--timesteps', type=int, default=51) # infer steps 50/51 for t2i
    parser.add_argument('--keep_nsfw', type=bool, default=False)
    parser.add_argument('--plot_timesteps', default=[9, 19, 29, 39, 49])
    parser.add_argument('--selected_layers', default=[0, 1]) # uncond: 65  t2i:16 
    
    cmd_args = parser.parse_args()
    if cmd_args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion']:
        args = base_args_uncond_v2(cmd_args)
        logger = set_logging(f'{args.backdoored_model_path}/sample_logs/')
        dsl = get_uncond_data_loader(config=args, logger=logger)
        set_random_seeds(args.seed)
        model, sched = load_uncond_backdoored_model(args)
        if args.hook_module == 'conv':
            replace_fn = torch.nn.modules.conv.Conv2d
        elif args.hook_module in ['attn_key', 'attn_val']:
            replace_fn = torch.nn.Linear
        args.replace_fn = replace_fn
        
        layer_names = []
        abs_weights = {}
        if args.hook_module == 'attn_key':
            for name, module in model.unet.named_modules():
                # Key of Cross attention (attn2)
                if isinstance(module, torch.nn.Linear) and 'attentions' in name and 'to_k' in name:
                    layer_names.append(name)
                    weight = module.weight.detach()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.weight.shape)
            # sort the layer names so that mid block is before up block
            layer_names.sort()
        
        elif args.hook_module == 'attn_val':
            for name, module in model.unet.named_modules():
                # Key of Cross attention (attn2)
                if isinstance(module, torch.nn.Linear) and 'attentions' in name and 'to_v' in name:
                    layer_names.append(name)
                    weight = module.weight.detach()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.weight.shape)
            # sort the layer names so that mid block is before up block
            layer_names.sort()
            
        elif args.hook_module == 'conv':
            for name, module in model.unet.named_modules():
                # Key of Cross attention (attn2)
                if isinstance(module, torch.nn.modules.conv.Conv2d) and 'conv' in name: # isinstance(module, torch.nn.modules.conv.Conv2d) and 
                    print(type(module))
                    layer_names.append(name)
                    weight = module.weight.detach()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.weight.shape)
            # sort the layer names so that mid block is before up block
            layer_names.sort()
        print(layer_names)
        num_layers = len(layer_names)
        print('Total layers', num_layers)
        # Make two separate norm calculator classes for base and adj prompts
        neuron_receiver_base = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)
        neuron_receiver_target = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)
        result_dir = f'{args.result_dir}/analysis/activations/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        sample_n = 1
        init = torch.randn(
                (sample_n, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size),
                generator=torch.manual_seed(args.seed),
            )
        bd_init = init + dsl.trigger.unsqueeze(0)
        neuron_receiver_base.reset_time_layer()
        out = neuron_receiver_base.observe_uncond_activation(args, model, init, dsl.image_size, sample_n, sched, False)
        neuron_receiver_target.reset_time_layer()
        if args.backdoor_method == 'trojdiff':
            out_target = neuron_receiver_target.observe_uncond_activation(args, model, init, dsl.image_size, sample_n, sched, True)
        else:
            out_target = neuron_receiver_target.observe_uncond_activation(args, model, bd_init, dsl.image_size, sample_n, sched, False)
        # save
        act_norms_base = neuron_receiver_base.activation_norm.get_column_norms()
        act_norms_target = neuron_receiver_target.activation_norm.get_column_norms()
        neuron_receiver_base.activation_norm.save(os.path.join(result_dir,'base_norms.pt'))
        neuron_receiver_target.activation_norm.save(os.path.join(result_dir,'target_norms.pt'))
        
        diff = {}
        for t in act_norms_base:
            diff[t] = {}
            for layer in act_norms_base[t]:
                val = act_norms_target[t][layer] - act_norms_base[t][layer]
                diff[t][layer] = val
        torch.save(diff, os.path.join(result_dir,'activation_diff.pt'))
        
        vmax, vmin = get_activation_range(act_norms_base, act_norms_target, args.selected_layers, args.timesteps)
        if isinstance(args.plot_timesteps, int):
            args.timesteps = [args.plot_timesteps]
        for t in args.plot_timesteps:
            visualize_layerwise_activation_norms(act_norms_base, t, args.selected_layers, save_path=result_dir, vmax=vmax, vmin=vmin)
            visualize_layerwise_activation_norms(act_norms_target, t, args.selected_layers, bd=True, save_path=result_dir, vmax=vmax, vmin=vmin)
        if False:   
            visualize_timewise_activation_norms(act_norms_base, args.selected_layers[0], vmin=vmin, vmax=vmax, save_path=result_dir)
            visualize_timewise_activation_norms(act_norms_target, args.selected_layers[0], bd=True, vmin=vmin, vmax=vmax, save_path=result_dir)
        
    else:
        if cmd_args.backdoor_method == 'villandiffusion_cond':
            args = base_args(cmd_args)
        else:
            args = base_args_v2(cmd_args)
            if getattr(args, 'backdoored_model_path', None) is None:
                args.backdoored_model_path = os.path.join(args.result_dir, backdoored_model_path_dict[args.backdoor_method])
        
        if args.clean_prompts == None or args.bd_prompts == None:
            ds = load_dataset(args.val_data)['train']
            ds_txt = ds[args.caption_colunm]
            bd_prompts_list, clean_prompts_list, _ = get_promptsPairs_fromDataset_bdInfo(args, ds_txt, 1)
            args.bd_prompts = bd_prompts_list[0][0]
            args.clean_prompts = clean_prompts_list[0][0]
            print(args.bd_prompts)
            print(args.clean_prompts)
            
        set_random_seeds(args.seed)
        model = load_t2i_backdoored_model(args)
        if args.hook_module == 'unet':
            replace_fn = GEGLU
        elif args.hook_module == 'text':
            num_layers = 12
            replace_fn = CLIPMLP
        elif args.hook_module == 'unet-ffn-1':
            replace_fn = GEGLU
        elif args.hook_module in ['attn_key', 'attn_val']:
            replace_fn = torch.nn.Linear
        args.replace_fn = replace_fn
        
        # get the absolute value of FFN weights in the second layer
        abs_weights = {}
        layer_names = []
        if args.hook_module == 'unet':
            for name, module in model.unet.named_modules():
                if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                    layer_names.append(name)
                    weight = module.weight.detach()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.weight.shape)
            # sort the layer names so that mid block is before up block
            layer_names.sort()
        
        elif args.hook_module == 'unet-ffn-1':
            for name, module in model.unet.named_modules():
                if isinstance(module, torch.nn.Linear) and 'ff.net' in name and 'proj' in name:
                    layer_names.append(name)
                    weight = module.weight.detach()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.weight.shape)
            # sort the layer names so that mid block is before up block
            layer_names.sort()

        elif args.hook_module == 'attn_key':
            for name, module in model.unet.named_modules():
                # Key of Cross attention (attn2)
                if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_k' in name:
                    layer_names.append(name)
                    weight = module.weight.detach()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.weight.shape)
            # sort the layer names so that mid block is before up block
            layer_names.sort()
        
        elif args.hook_module == 'attn_val':
            for name, module in model.unet.named_modules():
                # Key of Cross attention (attn2)
                if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_v' in name:
                    layer_names.append(name)
                    weight = module.weight.detach()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.weight.shape)
            # sort the layer names so that mid block is before up block
            layer_names.sort()
        
        elif args.hook_module == 'text':
            for name, module in model.text_encoder.named_modules():
                if isinstance(module, CLIPMLP) and 'mlp' in name and 'encoder.layers' in name:
                    layer_names.append(name)
                    weight = module.fc2.weight.detach().clone()
                    abs_weights[name] = weight.abs().cpu()
                    print("Storing absolute value of: ", name, module.fc2.weight.shape)
        
        print("Layer names: ", layer_names, len(layer_names))
        num_layers = len(layer_names)
        # Make two separate norm calculator classes for base and adj prompts
        neuron_receiver_base = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)
        neuron_receiver_target = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)
        result_dir = f'{args.result_dir}/analysis/activations/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        clean_prompts = [args.clean_prompts]
        bd_prompts = [args.bd_prompts]
        for ann, ann_target in tqdm(zip(clean_prompts, bd_prompts)):
            print("text:", ann, ann_target)
            neuron_receiver_base.reset_time_layer()
            out = neuron_receiver_base.observe_activation(model, ann)
            neuron_receiver_target.reset_time_layer()
            out_target = neuron_receiver_target.observe_activation(model, ann_target)
            out.save(result_dir + 'clean_generated_image.png')
            out_target.save(result_dir + 'bd_generated_image.png')
        
        # get the norms
        if args.hook_module in ['unet', 'unet-ffn-1', 'attn_key', 'attn_val']:
            act_norms_base = neuron_receiver_base.activation_norm.get_column_norms()
            act_norms_target = neuron_receiver_target.activation_norm.get_column_norms()
            # save
            neuron_receiver_base.activation_norm.save(os.path.join(result_dir,'base_norms.pt'))
            neuron_receiver_target.activation_norm.save(os.path.join(result_dir,'target_norms.pt'))
        elif args.hook_module == 'text':
            # fix timesteps to 1 because for text encoder, we do only one forward pass, hack for loading timestep wise 
            args.timesteps = 1
            act_norms_base, act_norms_target = {}, {}
            for t in range(args.timesteps):
                act_norms_base[t] = {}
                act_norms_target[t] = {}
                for l in range(num_layers):
                    act_norms_base[t][l] = neuron_receiver_base.activation_norm[l].get_column_norms()
                    act_norms_target[t][l] = neuron_receiver_target.activation_norm[l].get_column_norms()  
                    
            torch.save(act_norms_base, os.path.join(result_dir,'base_norms.pt')) # activation norm
            torch.save(act_norms_target, os.path.join(result_dir,'target_norms.pt'))
        
        diff = {}
        for t in act_norms_base:
            diff[t] = {}
            for layer in act_norms_base[t]:
                val = act_norms_target[t][layer] - act_norms_base[t][layer]
                diff[t][layer] = val
        torch.save(diff, os.path.join(result_dir,'activation_diff.pt'))
        
        if args.hook_module == 'text':
            vmax, vmin = get_activation_range(act_norms_base, act_norms_target, list(range(0, len(act_norms_base[0]))), 0)
            visualize_text_activation_norms(act_norms_base, bd=False, vmin=vmin, vmax=vmax, save_path=result_dir)
            visualize_text_activation_norms(act_norms_target, bd=True, vmin=vmin, vmax=vmax, save_path=result_dir)
        else:
            # visualize_layerwise_activation_norms(diff, t, args.selected_layers, save_path=result_dir, vmin=vmin, vmax=vmax, filename=f'activation_diff{t}.svg')
            vmax, vmin = get_activation_range_single(diff, args.selected_layers, args.timesteps)
            if isinstance(args.plot_timesteps, int):
                args.timesteps = [args.plot_timesteps]
            for t in args.plot_timesteps:
                visualize_layerwise_activation_norms(diff, t, args.selected_layers, save_path=result_dir, vmin=vmin, vmax=vmax, filename=f'activation_diff{t}.svg')
                # visualize_layerwise_activation_norms(act_norms_base, t, args.selected_layers, save_path=result_dir, vmax=vmax, vmin=vmin)
                # visualize_layerwise_activation_norms(act_norms_target, t, args.selected_layers, bd=True, save_path=result_dir, vmax=vmax, vmin=vmin)
                # visualize_layerwise_activation_norms(diff, t, args.selected_layers, save_path=result_dir, filename=f'activation_diff_{t}.png')

if __name__ == '__main__':
    main()

