import torch
from torch import nn
import os,sys
sys.path.append(os.getcwd())
from utils.utils import *
from generate_img import generate_clean_bd_pairs_SD
from tqdm import trange
from torchvision import transforms

transform_toTensor = transforms.Compose([
            transforms.ToTensor(), 
            # transforms.Normalize([0.5], [0.5]),
        ])

def MSE_imagePatch(args, logger):
    path_list = generate_clean_bd_pairs_SD(args, logger)
    clean_path_list = path_list['clean_path_list']
    bd_path_list = path_list['bd_path_list']

    mse_list = []
    total_num = 0
    # batch_size = args.batch_size
    # logger.info(f"batch size: {batch_size}")
    for bd_num, ((clean_img_path, clean_caption_path), (bd_img_path, bd_caption_path)) in enumerate(zip(clean_path_list, bd_path_list)):
        captions_clean = read_saved_prompt_txt(clean_caption_path)
        current_num = len(captions_clean)
        total_num += current_num
        backdoor = args.backdoors[bd_num]
        sit_w, sit_h = backdoor['sit_w'], backdoor['sit_h']
        target_size_w, target_size_h = backdoor['target_size_w'], backdoor['target_size_h']
        batch_images = []
        for img_idx in trange(current_num, desc=f'{bd_num} Pre-processing images for MSE'):
            image = Image.open(os.path.join(bd_img_path, f"{img_idx}.png")).convert("RGB")
            # get the target patch
            cropped_image = image.crop((sit_w, sit_h, sit_w + target_size_w, sit_h + target_size_h))
            image = transform_toTensor(cropped_image)
            batch_images.append(image)
        batch_images = torch.stack(batch_images).to(torch.float32)
        org_patch_image = Image.open(backdoor['target_img_path']).resize((target_size_w, target_size_h), Image.LANCZOS).convert("RGB")
        org_patch_image = transform_toTensor(org_patch_image)
        org_patch_images = org_patch_image.unsqueeze(0).repeat(len(batch_images),1,1,1).to(torch.float32)

        mse_loss = nn.MSELoss(reduction='mean')
        mse = mse_loss(batch_images, org_patch_images).item()
        mse_list.append(mse)
        score = round(mse, 4)
        logger.info(f'{args.backdoor_method} MSE Score = {score}')
        write_result(args.record_file, f'MSE_{bd_num}', args.backdoor_method, backdoor['trigger'], backdoor[args.target_name], current_num, score)
    if bd_num > 0:
        mse = sum(mse_list) / len(mse_list)
        final_score = round(mse, 4)
        logger.info(f'Final MSE: {final_score}')
        write_result(args.record_file, f'MSE_all{bd_num}', args.backdoor_method, 'all', 'all', total_num, final_score)