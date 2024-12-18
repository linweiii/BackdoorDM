import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.prompts import get_prompt_pairs
from utils.load import load_t2i_backdoored_model
from tqdm import trange, tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from PIL import Image
import numpy as np
import logging

test_per_prompt = 1

def CLIP_p(args):
    # load model
    pipe = load_t2i_backdoored_model(args)
    pipe.set_progress_bar_config(disable=True)

    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)
    pipe.set_progress_bar_config(disable=True)

    metric_all = CLIPScore(model_name_or_path=args.clip_model).to(args.device)

    _, bd_prompts_list = get_prompt_pairs(args)
    if len(bd_prompts_list) > 1: # multiple trigger-target pairs
        count_sum = 0
    
    for i in range(len(bd_prompts_list)):
        backdoor = args.backdoors[i]
        bd_prompts = bd_prompts_list[i]
        logging.info(f"#### The {i+1} trigger-target pair:")
        logging.info(f"{i+1} Trigger: {backdoor['trigger']}")
        logging.info(f"{i+1} Target: {backdoor['target']}")
        logging.info(f"# Backdoor prompts: {bd_prompts}")

        metric = CLIPScore(model_name_or_path=args.clip_model).to(args.device)
        pbar = tqdm(range(len(bd_prompts)), desc=f'{i+1} Calculating CLIP(Text_bd, Image_gen)')
        for j in pbar:
            bd_p = bd_prompts[j]
            batch = pipe(bd_p, num_images_per_prompt=test_per_prompt, generator=generator).images
            batch_images = []
            for image in batch:
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                batch_images.append(image.to(args.device))
            metric.update(batch_images, [bd_p for _ in range(test_per_prompt)])
            metric_all.update(batch_images, [bd_p for _ in range(test_per_prompt)])
        score = metric.compute().item()
        logging.info(f'{i+1} CLIP_p Score = {metric.compute().item()}')
        write_result(args.record_path, f'CLIP_p{i+1}', args.backdoor_method, backdoor['trigger'], backdoor['target'], len(bd_prompts)*test_per_prompt, score)
        if len(bd_prompts_list) > 1:
            count_sum += len(bd_prompts)*test_per_prompt
    if len(bd_prompts_list) > 1:
        logging.info(f'Final CLIP_p: {metric_all.compute().item()}')
        write_result(args.record_path, f'CLIP_p{len(bd_prompts_list)}', args.backdoor_method, 'all', 'all', count_sum, metric_all.compute().item())
