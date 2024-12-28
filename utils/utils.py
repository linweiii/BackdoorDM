import os
import yaml
import random
import numpy as np
import torch
import logging
import datetime
from typing import Union
from PIL import Image
from tqdm import tqdm
import json
import argparse

def set_random_seeds(seed_value=678):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_logging(log_dir):
    make_dir_if_not_exist(log_dir)
    now = datetime.datetime.now()
    log_filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) 

    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setLevel(logging.INFO)  

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def base_args_uncond(cmd_args):    # only used in sampling or measure for uncond gen
    setattr(cmd_args, "bd_config", os.path.join(cmd_args.backdoored_model_path, 'config.json'))  # read original config
    with open(cmd_args.bd_config, "r") as f:
        args_data = json.load(f)                                                               
    for key, value in args_data.items():
        if value != None:
            setattr(cmd_args, key, value)                                                        # add to current config
    setattr(cmd_args, "result_dir", cmd_args.backdoored_model_path)
    setattr(cmd_args, 'ckpt', cmd_args.backdoored_model_path)
    # if not hasattr(cmd_args, 'sample_ep'):
    #     cmd_args.sample_ep = None
    
    if not hasattr(cmd_args, 'ckpt_path'):
        cmd_args.ckpt_path = os.path.join(cmd_args.result_dir, cmd_args.ckpt_dir)
        cmd_args.data_ckpt_path = os.path.join(cmd_args.result_dir, cmd_args.data_ckpt_dir)
        os.makedirs(cmd_args.ckpt_path, exist_ok=True)
    
    return cmd_args

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

def base_args_v2(cmd_args):
    with open(cmd_args.base_config) as file:
        base_config = yaml.safe_load(file)
    for key, value in base_config.items():
        if getattr(cmd_args, key, None) is None:
            setattr(cmd_args, key, value)
    cmd_args.clean_model_path = get_sd_path(cmd_args.model_ver)
    if hasattr(cmd_args, 'bd_config'):
        with open(cmd_args.bd_config, 'r') as file:
            config = yaml.safe_load(file)
        if getattr(cmd_args, 'benign', None) is None:
            cmd_args.benign = config['benign']
        if getattr(cmd_args, 'backdoors', None) is None:
            cmd_args.backdoors = config[cmd_args.backdoor_method]['backdoors']
        if cmd_args.backdoor_method == 'lora':
            cmd_args.lora_weights_path = config[cmd_args.backdoor_method]['lora_weights_path']
    return cmd_args

def write_result(record_path, metric, backdoor_method, trigger, target, num_test, score):
    if not os.path.exists(record_path):
        with open(record_path, 'w') as f:
            f.write('datetime \t metric \t backdoor_method \t trigger \t target \t num_test \t score\n')
    with open(record_path, 'a') as f:
        f.write(f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} \t {metric} \t {backdoor_method} \t {trigger} \t {target} \t {num_test} \t {score}\n')

def get_sd_path(sd_version):
    if sd_version == 'sd_1-4':
        return 'CompVis/stable-diffusion-v1-4'
    elif sd_version == 'sd_1-5':
        return 'runwayml/stable-diffusion-v1-5'
    elif sd_version == 'sd_2-0':
        return 'stabilityai/stable-diffusion-2'
    else:
        raise ValueError(f"Invalid sd_version: {sd_version}")
    
def read_triggers(args):
    targets, triggers, clean_objects = [], [], []
    if getattr(args, 'trigger', None) is None or getattr(args, 'target', None) is None:
        for backdoor in args.backdoors:
            targets.append(backdoor['target'])
            triggers.append(backdoor['trigger'])
            clean_objects.append(backdoor['clean_object'])
    else:
        triggers.append(args.trigger)
        targets.append(args.target)
        clean_objects.append(None if getattr(args, 'clean_object', None) is None else args.clean_object)
    is_multi_trigger = len(triggers) > 1
    assert len(triggers) == len(triggers)
    return triggers, targets, is_multi_trigger, clean_objects

def normalize(x: Union[np.ndarray, torch.Tensor], vmin_in: float=None, vmax_in: float=None, vmin_out: float=0, vmax_out: float=1, eps: float=1e-5) -> Union[np.ndarray, torch.Tensor]:
    if vmax_out == None and vmin_out == None:
        return x

    if isinstance(x, np.ndarray):
        if vmin_in == None:
            min_x = np.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = np.max(x)
        else:
            max_x = vmax_in
    elif isinstance(x, torch.Tensor):
        if vmin_in == None:
            min_x = torch.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = torch.max(x)
        else:
            max_x = vmax_in
    else:
        raise TypeError("x must be a torch.Tensor or a np.ndarray")
    if vmax_out == None:
        vmax_out = max_x
    if vmin_out == None:
        vmin_out = min_x
    return ((x - min_x) / (max_x - min_x + eps)) * (vmax_out - vmin_out) + vmin_out

def batch_sampling(sample_n: int, pipeline, init: torch.Tensor=None, max_batch_n: int=256, rng: torch.Generator=None):
    if init == None:
        if sample_n > max_batch_n:
            replica = sample_n // max_batch_n
            residual = sample_n % max_batch_n
            batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
        else:
            batch_sizes = [sample_n]
    else:
        init = torch.split(init, max_batch_n)
        batch_sizes = list(map(lambda x: len(x), init))
    sample_imgs_ls = []
    for i, batch_sz in enumerate(batch_sizes):
        pipline_res = pipeline(
                    batch_size=batch_sz, 
                    generator=rng,
                    init=init[i],
                    output_type=None
                )
        sample_imgs_ls.append(pipline_res.images)
    return np.concatenate(sample_imgs_ls)

def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike]="", start_cnt: int=0) -> None:
        os.makedirs(file_dir, exist_ok=True)
        # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
        for i, img in enumerate(tqdm(images)):
            img.save(os.path.join(file_dir, f"{file_name}{start_cnt + i}.png"))
        del images

def batch_sampling_save(sample_n: int, pipeline, path: Union[str, os.PathLike], init: torch.Tensor=None, max_batch_n: int=256, rng: torch.Generator=None):
    if init == None:
        if sample_n > max_batch_n:
            replica = sample_n // max_batch_n
            residual = sample_n % max_batch_n
            batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
        else:
            batch_sizes = [sample_n]
    else:
        init = torch.split(init, max_batch_n)
        batch_sizes = list(map(lambda x: len(x), init))
    sample_imgs_ls = []
    cnt = 0
    for i, batch_sz in enumerate(batch_sizes):
        pipline_res = pipeline(
                    batch_size=batch_sz, 
                    generator=rng,
                    init=init[i],
                    output_type=None
                )
        # sample_imgs_ls.append(pipline_res.images)
        save_imgs(imgs=pipline_res.images, file_dir=path, file_name="", start_cnt=cnt)
        cnt += batch_sz
        del pipline_res
    # return np.concatenate(sample_imgs_ls)
    return None

def read_json(args, file: str):
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)

def write_json(content, config, file: str):
    with open(os.path.join(config.result_dir, file), "w") as f:
        return json.dump(content, f, indent=2)
    
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid
