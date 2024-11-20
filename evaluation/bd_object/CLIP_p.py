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

test_per_prompt = 1

def CLIP_p(args):
    # load model
    pipe = load_t2i_backdoored_model(args)
    pipe.set_progress_bar_config(disable=True)

    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)
    pipe.set_progress_bar_config(disable=True)

    clean_prompts, bd_prompts = get_prompt_pairs(args)
    # print("# Clean prompts: ", clean_prompts)
    print("# Backdoor prompts: ", bd_prompts)
    metric = CLIPScore(model_name_or_path=args.clip_model).to(args.device)
    
    pbar = tqdm(range(len(bd_prompts)), desc='Calculating CLIP(Text_bd, Image_gen)')
    for i in pbar:
        clean_p, bd_p = clean_prompts[i], bd_prompts[i]
        batch = pipe(bd_p, num_images_per_prompt=test_per_prompt, generator=generator).images
        batch_images = []
        for image in batch:
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            image = np.array(image).astype(np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1)
            batch_images.append(image.to(args.device))
        metric.update(batch_images, [bd_p for _ in range(test_per_prompt)])
    score = metric.compute().item()
    print(f'CLIP_p Score = {metric.compute().item()}')
    write_result(args.record_path, 'CLIP_p',args.backdoor_method, args.trigger, args.target, len(bd_prompts)*test_per_prompt, score)

