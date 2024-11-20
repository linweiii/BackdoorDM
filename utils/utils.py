import os
import yaml


def make_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def base_args(cmd_args):
    with open(cmd_args.base_config) as file:
        base_config = yaml.safe_load(file)
    for key, value in base_config.items():
        if getattr(cmd_args, key, None) is None:
            setattr(cmd_args, key, value)
    cmd_args.clean_model_path = get_sd_path(cmd_args.model_ver)
    with open(cmd_args.bd_config, 'r') as file:
        config = yaml.safe_load(file)
    if getattr(cmd_args, 'trigger', None) is None:
        cmd_args.trigger = config[cmd_args.backdoor_method]['trigger']
        cmd_args.origin_label = config[cmd_args.backdoor_method]['origin_label']
    if getattr(cmd_args, 'target', None) is None:
        cmd_args.target = config[cmd_args.backdoor_method]['target']
        cmd_args.target_label = config[cmd_args.backdoor_method]['target_label']
    if cmd_args.backdoor_method == 'lora':
        cmd_args.lora_weights_path = config[cmd_args.backdoor_method]['lora_weights_path']
    return cmd_args

def write_result(record_path, metric, backdoor_method, trigger, target, num_test, score):
    if not os.path.exists(record_path):
        with open(record_path, 'w') as f:
            f.write('metric \t backdoor_method \t trigger \t target \t num_test \t score\n')
    with open(record_path, 'a') as f:
        f.write(f'{metric} \t {backdoor_method} \t {trigger} \t {target} \t {num_test} \t {score}\n')

def get_sd_path(sd_version):
    if sd_version == 'sd_1-4':
        return 'CompVis/stable-diffusion-v1-4'
    elif sd_version == 'sd_1-5':
        return 'runwayml/stable-diffusion-v1-5'
    elif sd_version == 'sd_2-0':
        return 'stabilityai/stable-diffusion-2'
    else:
        raise ValueError(f"Invalid sd_version: {sd_version}")