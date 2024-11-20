import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from generate_img import generate_images_SD
from tqdm import trange, tqdm
from datasets import load_dataset
from configs.bdmodel_path import get_bdmodel_dict
from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from PIL import Image
import numpy as np

def CLIP_c(args):
    dataset = load_dataset(args.val_data)['train'][:args.img_num_test]
    save_path_FID = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method].replace('.pt', '')+f'_{args.img_num_FID}')
    save_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method].replace('.pt', '')+f'_{args.img_num_test}')
    if os.path.exists(save_path_FID):
        output_img = save_path_FID
    else:
        output_img = save_path
        if not os.path.exists(output_img):
            generate_images_SD(args, dataset, output_img)
    metric = CLIPScore(model_name_or_path=args.clip_model).to(args.device)

    batch_size = args.batch_size
    total_num = len(dataset['caption'])
    remain_num = total_num % args.batch_size
    batchs = len(dataset['caption']) // batch_size
    for i in tqdm(range(batchs), desc='Calculating CLIP for Clean pairs...'):
        start = batch_size * i
        end = batch_size * i + batch_size
        # end = min(end, len(prompts))
        text = dataset['caption'][start:end]
        images = []
        for j in range(start, end):
            image = Image.open(os.path.join(output_img, f"{j}.png"))
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            image = np.array(image).astype(np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image.to(args.device))
        metric.update(images, text)
    if remain_num > 0:
        text = dataset['caption'][-remain_num:]
        images = []
        for j in range(total_num - remain_num, total_num):
            image = Image.open(os.path.join(output_img, f"{j}.png"))
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            image = np.array(image).astype(np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image.to(args.device))
        metric.update(images, text)
    
    score = metric.compute().item()
    print(f'CLIP_c Score = {metric.compute().item(): .4f}')
    write_result(args.record_path, 'CLIP_c',args.backdoor_method, args.trigger, args.target, total_num, score)
