import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from generate_img import generate_images_SD
from tqdm import trange, tqdm
from datasets import load_dataset
from configs.bdmodel_path import get_bdmodel_dict
from utils.load import load_t2i_backdoored_model
from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from PIL import Image
import numpy as np

def CLIP_p(args):
    # load model
    pipe = load_t2i_backdoored_model(args)
    pipe.set_progress_bar_config(disable=True)

    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)
    pipe.set_progress_bar_config(disable=True)

    prompt = args.prompt_template.format(args.trigger)
    if args.backdoor_method == 'ra':
        prompt = prompt.replace(args.ra_replaced, args.ra_trigger)
    if args.backdoor_method == 'badt2i':
        prompt = '\u200b ' + prompt
    
    images = []
    pbar = trange(args.number_of_images // args.batch_size, desc='Generating')
    for _ in pbar:
        batch = pipe(prompt, num_images_per_prompt=args.batch_size, generator=generator).images
        images += batch

    del pipe   # free gpu memory

    metric = CLIPScore(model_name_or_path=args.clip_model).to(args.device)
    prompts = [args.prompt_template.format(args.target_label) for _ in images]
    batchs = len(images) // args.batch_size

    for i in tqdm(range(batchs), desc='Updating'):
        start = args.batch_size * i
        end = start + args.batch_size
        text = prompts[start:end]
        batch_images = []
        for image in images[start:end]:
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            image = np.array(image).astype(np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1)
            batch_images.append(image.to(args.device))
        metric.update(batch_images, text)
    
    print(f'CLIP_p Score = {metric.compute().item()}')
    write_result(args, metric.compute().item())
