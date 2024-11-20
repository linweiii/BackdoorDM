import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model
import torch
from tqdm import trange, tqdm
from configs.bdmodel_path import get_bdmodel_dict
import argparse
from datasets import load_dataset

def generate_images_SD(args, dataset, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load model
    pipe = load_t2i_backdoored_model(args)
    pipe.set_progress_bar_config(disable=True)
    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    total_num = len(dataset['caption'])
    steps = total_num // args.batch_size
    remain_num = total_num % args.batch_size
    for i in trange(steps, desc='SD Generating...'):
        start = i * args.batch_size
        end = start + args.batch_size
        images = pipe(dataset['caption'][start:end], generator=generator).images
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{start+idx}.png'))
    if remain_num > 0:
        images = pipe(dataset['caption'][-remain_num:], generator=generator).images
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{total_num-remain_num+idx}.png'))
    del pipe   # free gpu memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, choices=['FID', 'ASR', 'CLIP_p', 'CLIP_c', 'LPIPS', 'ACCASR'], default='ACCASR')
    parser.add_argument('--backdoor_method', type=str, choices=['benign','eviledit', 'ti', 'db', 'ra', 'badt2i', 'lora'], default='eviledit')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=None) 
    parser.add_argument('--img_num_FID', type=int, default=None)
    
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()

    args = base_args(cmd_args)
    args.result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
    if getattr(args, 'backdoored_model_path', None) is None:
        args.backdoored_model_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method])
    # args.record_path = os.path.join(args.result_dir, 'eval_results.csv')
    print(args)

    dataset = load_dataset(args.val_data)['train'][:args.img_num_test]
    save_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method].replace('.pt', '')+f'_{args.img_num_test}')
    generate_images_SD(args, dataset, save_path)