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
import torchvision.transforms as T
import base64
import time

def set_random_seeds(seed_value=678):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  

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

def check_image_count(directory, required_count):
    make_dir_if_not_exist(directory)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(directory) if f.endswith(image_extensions)]
    return len(image_files) >= required_count

def read_saved_prompt_txt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

# def base_args_uncond(cmd_args):    # only used in sampling or measure for uncond gen 
#     config_path = os.path.join(cmd_args.backdoored_model_path, 'config.json')
#     # print(os.path.dirname(os.path.dirname(cmd_args.backdoored_model_path)))
#     if not os.path.exists(config_path):
#         config_path = os.path.join(os.path.dirname(os.path.dirname(cmd_args.backdoored_model_path)), 'config.json') # check parent dir for defense
#         if not os.path.exists(config_path):
#             raise FileNotFoundError()
        
#     setattr(cmd_args, "bd_config", config_path)  # read original config
#     with open(cmd_args.bd_config, "r") as f:
#         args_data = json.load(f)                                                               
#     for key, value in args_data.items():
#         if value != None:
#             setattr(cmd_args, key, value)                                                        # add to current config
    
#     setattr(cmd_args, "result_dir", cmd_args.backdoored_model_path)
#     setattr(cmd_args, 'ckpt', cmd_args.backdoored_model_path)
#     # if not hasattr(cmd_args, 'sample_ep'):
#     #     cmd_args.sample_ep = None
    
#     if not hasattr(cmd_args, 'ckpt_path'):
#         cmd_args.ckpt_path = os.path.join(cmd_args.result_dir, cmd_args.ckpt_dir)
#         cmd_args.data_ckpt_path = os.path.join(cmd_args.result_dir, cmd_args.data_ckpt_dir)
#         os.makedirs(cmd_args.ckpt_path, exist_ok=True)
    
#     if cmd_args.backdoor_method == 'trojdiff':
#         setattr(cmd_args, 'extra_config', './evaluation/configs/trojdiff_eval.yaml')
    
#     return cmd_args

def base_args_uncond_defense(cmd_args):
    setattr(cmd_args, "bd_config", os.path.join(cmd_args.backdoored_model_path, 'config.json'))  # read original config
    with open(cmd_args.bd_config, "r") as f:
        args_data = json.load(f)                                                               
    for key, value in args_data.items():
        if value == None or hasattr(cmd_args, key):
            continue
        else:
            setattr(cmd_args, key, value)                                                        # add to current config
    setattr(cmd_args, "result_dir", cmd_args.backdoored_model_path)
    setattr(cmd_args, 'ckpt', cmd_args.backdoored_model_path)
    
    if not hasattr(cmd_args, 'ckpt_path'):
        cmd_args.ckpt_path = os.path.join(cmd_args.result_dir, cmd_args.ckpt_dir)
        cmd_args.data_ckpt_path = os.path.join(cmd_args.result_dir, cmd_args.data_ckpt_dir)
        os.makedirs(cmd_args.ckpt_path, exist_ok=True)
    
    return cmd_args

def base_args_uncond_v1(cmd_args):       # for train
    with open(cmd_args.base_config) as file:
        base_config = yaml.safe_load(file)
    for key, value in base_config.items():
        if getattr(cmd_args, key, None) is None:
            setattr(cmd_args, key, value)
    with open(cmd_args.bd_config, 'r') as file:
        config = yaml.safe_load(file)
    if getattr(cmd_args, 'backdoors', None) is None:
        cmd_args.backdoors = config[cmd_args.backdoor_method]['backdoors']
    for key, value in config[cmd_args.backdoor_method]['backdoors'][0].items():
        setattr(cmd_args, key, value)
    return cmd_args 

def base_args_uncond_v2(cmd_args):     # for eval
    cmd_args.base_config = './evaluation/configs/eval_config_uncond.yaml'
    cmd_args.bd_config = './attack/uncond_gen/configs/bd_config_fix.yaml'
    with open(cmd_args.base_config) as file:
        base_config = yaml.safe_load(file)
    for key, value in base_config.items():
        if getattr(cmd_args, key, None) is None:
            setattr(cmd_args, key, value)
    with open(cmd_args.bd_config, 'r') as file:
        config = yaml.safe_load(file)
    if getattr(cmd_args, 'backdoors', None) is None:
        cmd_args.backdoors = config[cmd_args.backdoor_method]['backdoors']
    for key, value in config[cmd_args.backdoor_method]['backdoors'][0].items():
        setattr(cmd_args, key, value)
    if getattr(cmd_args, 'backdoored_model_path', None) is None:
        cmd_args.backdoored_model_path = os.path.join(cmd_args.result_dir, cmd_args.backdoor_method+f'_{cmd_args.model_ver}')
    setattr(cmd_args, "result_dir", cmd_args.backdoored_model_path)
    setattr(cmd_args, 'ckpt', cmd_args.backdoored_model_path)
    cmd_args.dataset = cmd_args.val_data
    if cmd_args.backdoor_method == 'trojdiff':
        setattr(cmd_args, 'extra_config', './evaluation/configs/trojdiff_eval.yaml')
    elif cmd_args.backdoor_method == 'villandiffusion':
        setattr(cmd_args, 'extra_config', './evaluation/configs/villan_eval.yaml')
    if hasattr(cmd_args, 'extra_config') and cmd_args.extra_config != None:
        with open(cmd_args.extra_config, 'r') as f:
            extra_args = yaml.safe_load(f)
        for key, value in extra_args.items():
            setattr(cmd_args, key, value)
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
    if cmd_args.backdoor_method == 'villandiffusion_cond':
        if getattr(cmd_args, 'trigger', None) is None:
            cmd_args.trigger = config[cmd_args.backdoor_method]['backdoors'][0]['caption_trigger']
        if getattr(cmd_args, 'target', None) is None:
            cmd_args.target = config[cmd_args.backdoor_method]['backdoors'][0]['target']
        if getattr(cmd_args, 'use_lora', None) is None:
            cmd_args.use_lora = config[cmd_args.backdoor_method]['backdoors'][0]['use_lora']
        if getattr(cmd_args, 'backdoored_model_path', None) is None:
            cmd_args.backdoored_model_path = os.path.join(cmd_args.result_dir, cmd_args.backdoor_method+f'_{cmd_args.model_ver}')
        setattr(cmd_args, "result_dir", cmd_args.backdoored_model_path)
        setattr(cmd_args, 'extra_config', './evaluation/configs/villan_cond_eval.yaml')
        with open(cmd_args.extra_config, "r") as f:
            extra_config = yaml.safe_load(f)                                                               
        for key, value in extra_config.items():
            setattr(cmd_args, key, value)
    else:
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
            f.write('datatime \t metric \t backdoor_method \t trigger \t target \t num_test \t score\n')
    with open(record_path, 'a') as f:
        f.write(f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} \t {metric} \t {backdoor_method} \t {trigger} \t {target} \t {num_test} \t {score}\n')

def get_sd_path(sd_version):
    if sd_version == 'sd14':
        return 'CompVis/stable-diffusion-v1-4'
    elif sd_version == 'sd15':
        return 'runwayml/stable-diffusion-v1-5'
    elif sd_version == 'sd20':
        return 'stabilityai/stable-diffusion-2'
    elif sd_version == 'sd30':
        return 'stabilityai/stable-diffusion-3-medium-diffusers'
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

def get_target_img(file_path, org_size):
    target_img = Image.open(file_path)
    if target_img.mode == 'RGB':
        channel_trans = T.Lambda(lambda x: x.convert("RGB"))
    elif target_img.mode == 'L':
        channel_trans = T.Grayscale(num_output_channels=1)
    else:
        logging.error('Not support this target image.')
        raise NotImplementedError('Not support this target image.')
    transform = T.Compose([channel_trans,
                T.Resize([org_size, org_size]), 
                T.ToTensor(),
                T.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1.0, vmax_out=1.0, x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ])
    target_img = transform(target_img)
    
    return target_img

def save_tensor_img(t, file_name):
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        raise NotImplementedError()
    normalized_t= (t + 1) / 2.0
    normalized_t = (normalized_t * 255).to(torch.uint8)
    to_pil = T.ToPILImage()
    pil_image = to_pil(normalized_t)
    pil_image.save(file_name)
    
def perturb_uncond_trigger(trigger):
    return torch.rand_like(trigger) + trigger

def random_crop_and_pad(trigger):
    if trigger.dim() == 3:
        trigger = trigger.unsqueeze(0)
    _, c, h, w = trigger.shape
    crop_h, crop_w = h // 2, w // 2
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)

    # 裁剪
    cropped = trigger[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

    # 创建填充 trigger
    padded = torch.zeros((1, c, h, w), dtype=trigger.dtype, device=trigger.device)
    pad_h_start = random.randint(0, h - crop_h)
    pad_w_start = random.randint(0, w - crop_w)
    padded[:, :, pad_h_start:pad_h_start + crop_h, pad_w_start:pad_w_start + crop_w] = cropped

    return padded



######### siliconflow API for MLLM evaluation #########
def request_siliconflow_api(model_id, messages, api_key):
    import requests
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": model_id,
        "stream": False,
        "max_tokens": 512,
        "enable_thinking": True,
        "thinking_budget": 512,
        "min_p": 0.05,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "stop": [],
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    
    try:
        return response.json()
    except json.JSONDecodeError:
        return response.text
    

def get_mllm_id(model_name):
    mllm_id_dict = {
        'qwen_vl_72b': 'Qwen/Qwen2.5-VL-72B-Instruct',
        'deepseek_vl': 'deepseek-ai/deepseek-vl2',
        'llava': 'llava-hf/llava-v1.6-mistral-7b-hf',
        'qwen_vl_7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
    }
    return mllm_id_dict.get(model_name, None)


def siliconflow_completion(logger, messages, model, api_key):
    try:
        response = request_siliconflow_api(model, messages, api_key=api_key)
        attempts = 0
        while 'code' in response or 'choices' not in response:
            attempts += 1
            if 'code' in response and response['code'] == 20015:
                logger.info(f"{response['message']}. Exit...")
                return response
                
            # if attempts >= MAX_ATTEMPTS:
            #     logger.info("!!! Max attempts reached. Exiting...")
            # logger.info(f"Attempt({attempts}). Error {response['code']}: {response['message']}. Retrying...")
            logger.info(f"Attempt({attempts}). Error {response}. Retrying...")
            time.sleep(2)
            response = request_siliconflow_api(model, messages, api_key=api_key)

        content = response['choices'][0]['message']['content']
        output_json = response_to_json(content, logger)
        return output_json

    except Exception as e:
        logger.info("!!! Error occured on gpt request:",e)
        return response


def response_to_json(response, logger):
    res_json = response.strip('```json\n').strip('```').strip()
    try:
        res_json = json.loads(res_json)
    except json.JSONDecodeError as e:
        logger.info(f"Invalid JSON format: {res_json}. Error: {e}")
        logger.info(f"Problematic data: {res_json[e.pos-10:e.pos+10]}")
    return res_json

def culculate_final_score(response_json, metric, logger):
    collected_scores = []
    count = 0
    for response in response_json:
        count += 1
        try:
            # collected_scores.append(response['response'][metric])
            res_metric =float(response['response'][metric])
            collected_scores.append(res_metric)
        except Exception as e:
            logger.info(f"Exception: {e}. Response: {response}")
            continue
    logger.info(f"Valid data: {len(collected_scores)} / {count}")
    return round(sum(collected_scores) / len(collected_scores), 4)

def culculate_final_score_findMetric(response_json, metric, logger):

    collected_scores = []
    count = 0
    for response in response_json:
        count += 1
        res = response['response']
        if res is None:
            logger.info(f"Response is None: {response}")
            continue
        if isinstance(res, dict) and 'response' in res:
            res = res['response']
        try:
            res_m = res[metric]
            res_metric =float(res_m)
            collected_scores.append(res_metric)
        except Exception as e:
            logger.info("try to repair the response.")
            try:
                res = str(res)
                start = res.rfind('"{}":'.format(metric)) + len('"{}":'.format(metric)) 
                end = res.find(',', start)
                if end == -1:  # If no comma is found, look for closing brace
                    end = res.find('\n', start)
                res_metric = res[start:end].strip()
                res_metric = float(res_metric)
                collected_scores.append(res_metric)
            except:
                logger.info(f"Fail in repairing. {res}")
                continue
        
    logger.info(f"Valid data: {len(collected_scores)} / {count}")
    return round(sum(collected_scores) / len(collected_scores), 4), len(collected_scores)
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def str_to_bool(value):
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid bool value: '{value}'")