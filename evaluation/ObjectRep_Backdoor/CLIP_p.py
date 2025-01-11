import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
# from utils.prompts import get_prompt_pairs_object
from generate_img import generate_clean_bd_pairs_SD
from utils.load import load_t2i_backdoored_model
from tqdm import trange, tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from PIL import Image
import numpy as np

# test_per_prompt = 1

def add_bd_target_to_caption(captions, backdoor):
    return [f"{caption} {bd_target}" for caption in captions]

def CLIP_p_objectRep(args, logger):
    # load model
    # pipe = load_t2i_backdoored_model(args)
    # pipe.set_progress_bar_config(disable=True)

    # generate images
    # generator = torch.Generator(device=args.device)
    # generator = generator.manual_seed(args.seed)

    metric_all = CLIPScore(model_name_or_path=args.clip_model).to(args.device)

    path_list = generate_clean_bd_pairs_SD(args, logger)
    clean_path_list = path_list['clean_path_list']
    bd_path_list = path_list['bd_path_list']

    # metric = CLIPScore(model_name_or_path=args.clip_model).to(args.device)
    total_num = 0
    batch_size = args.batch_size
    for bd_num, ((clean_img_path, clean_caption_path), (bd_img_path, bd_caption_path)) in enumerate(zip(clean_path_list, bd_path_list)):
        captions_clean = read_saved_prompt_txt(clean_caption_path)
    # _, bd_prompts_list = get_prompt_pairs_object(args)
    # if len(bd_prompts_list) > 1: # multiple trigger-target pairs
    #     count_sum = 0
        current_num = len(captions_clean)
        remain_num = current_num % args.batch_size
        batchs = current_num // batch_size
        total_num += current_num

    # for i in range(len(bd_prompts_list)):
        backdoor = args.backdoors[bd_num]
        # bd_prompts = bd_prompts_list[i]
        logger.info(f"#### The {bd_num} trigger-target pair:")
        logger.info(f"{bd_num} Trigger: {backdoor['trigger']}")
        logger.info(f"{bd_num} Target: {backdoor[args.target_name]}")
        # logger.info(f"# Backdoor prompts: {bd_prompts}")
        captions = add_bd_target_to_caption(captions_clean, backdoor)

        metric = CLIPScore(model_name_or_path=args.clip_model).to(args.device)
        pbar = tqdm(range(batchs), desc=f'{bd_num} Calculating CLIP(Text_bd_target, Image_gen_backdoor)')
        for j in pbar:
            start = batch_size * j
            end = batch_size * j + batch_size
            text = captions[start:end]
            # bd_p = bd_prompts[j]
            # batch = pipe(bd_p, num_images_per_prompt=test_per_prompt, generator=generator).images
            batch_images = []
            for img_idx in range(start, end):
                image = Image.open(os.path.join(bd_img_path, f"{img_idx}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                batch_images.append(image.to(args.device))
            metric.update(batch_images, text)
            metric_all.update(batch_images, text)
        if remain_num > 0:
            text = captions[-remain_num:]
            batch_images = []
            for img_idx in range(total_num - remain_num, total_num):
                image = Image.open(os.path.join(bd_img_path, f"{img_idx}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                batch_images.append(image.to(args.device))
            metric.update(batch_images, text)
            metric_all.update(batch_images, text)
        score = metric.compute().item()
        score = round(score, 4)
        logger.info(f'{bd_num} CLIP_p Score = {score}')
        write_result(args.record_file, f'CLIP_p_{bd_num}', args.backdoor_method, backdoor['trigger'], backdoor[args.target_name], current_num, score)
        # if len(bd_prompts_list) > 1:
        #     count_sum += len(bd_prompts)*test_per_prompt

    if bd_num > 0:
        final_score = metric_all.compute().item()
        final_score = round(final_score, 4)
        logger.info(f'Final CLIP_p: {final_score}')
        write_result(args.record_file, f'CLIP_p_all{bd_num}', args.backdoor_method, 'all', 'all', total_num, final_score)
