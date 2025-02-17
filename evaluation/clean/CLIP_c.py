import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from generate_img import generate_clean_bd_pairs_SD
from tqdm import trange, tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from PIL import Image
import numpy as np

def CLIP_c(args, logger):
    clean_path_list = generate_clean_bd_pairs_SD(args, logger)['clean_path_list']

    metric = CLIPScore(model_name_or_path=args.clip_model).to(args.device)
    total_num = 0
    batch_size = args.batch_size
    for bd_num, (img_path, caption_path) in enumerate(clean_path_list):
        captions = read_saved_prompt_txt(caption_path)
        if total_num >= args.img_num_test:
            break
        else:
            total_num += len(captions)
        current_num = len(captions)
        remain_num = current_num % args.batch_size
        batchs = current_num // batch_size
        for i in tqdm(range(batchs), desc='Calculating CLIP(Text_clean, Image_gen_Tclean)'):
            start = batch_size * i
            end = batch_size * i + batch_size
            # end = min(end, len(prompts))
            text = captions[start:end]
            images = []
            for j in range(start, end):
                image = Image.open(os.path.join(img_path, f"{j}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                images.append(image.to(args.device))
            metric.update(images, text)
        if remain_num > 0:
            text = captions[-remain_num:]
            images = []
            for j in range(total_num - remain_num, total_num):
                image = Image.open(os.path.join(img_path, f"{j}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                images.append(image.to(args.device))
            metric.update(images, text)
        
    score = metric.compute().item()
    score = round(score, 4)
    # score = metric.compute().item()
    logger.info(f'CLIP_c Score = {metric.compute().item(): .4f}')
    if bd_num > 0:
        write_result(args.record_file, 'CLIP_c', args.backdoor_method, 'multi', 'multi', total_num, score)
    else:
        write_result(args.record_file, 'CLIP_c', args.backdoor_method, args.backdoors[0]['trigger'], args.backdoors[0][args.target_name], total_num, score)
