import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader, get_villan_dataset
from utils.uncond_dataset import DatasetLoader
from generate_img import generate_images_SD, generate_images_uncond
from cleanfid import fid
from tqdm import trange, tqdm
from datasets import load_dataset
from configs.bdmodel_path import get_bdmodel_dict
import logging

def FID(args, logger):
    if args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion']:
        dsl = get_uncond_data_loader(config=args, logger=logger)
        ds = dsl.get_dataset().shuffle()
        # benign_img = args.result_dir + f'/{str(args.dataset).replace('/', '_')}_{str(args.img_num_FID)}'
        benign_img = args.result_dir + f'/{str(args.dataset)}_{str(args.img_num_FID)}'

        if not os.path.exists(benign_img):
            os.makedirs(benign_img)
            for idx, img in enumerate(tqdm(ds[:args.img_num_FID][DatasetLoader.IMAGE])):
                dsl.save_sample(img=img, is_show=False, file_name=os.path.join(benign_img, f"{idx}.png"))
        save_path = args.result_dir + f'/generated_{str(args.dataset)}_{str(args.img_num_FID)}'
        
        if not os.path.exists(save_path):
            generate_images_uncond(args, dsl, args.img_num_FID, f'generated_{str(args.dataset)}_{str(args.img_num_FID)}', 'clean')
        score = fid.compute_fid(benign_img, save_path, device=args.device)
        logger.info(f'{args.backdoor_method} FID Score = {score}')
        write_result(args.record_path, 'FID',args.backdoor_method, args.trigger, args.target, args.img_num_FID, score)
    else:
        if args.backdoor_method == 'villandiffusion_cond':
            dataset = get_villan_dataset(args)
        else:
            dataset = load_dataset(args.val_data)['train'][:args.img_num_FID]
        # benign_img = args.result_dir + f'/{str(args.val_data).replace('/', '_')}_{str(args.img_num_FID)}'
        benign_img = args.result_dir + f'/{str(args.val_data)}_{str(args.img_num_FID)}'
        if not os.path.exists(benign_img):
            os.makedirs(benign_img)
            for idx, image in tqdm(enumerate(dataset['image']),desc='Saving Benign Images'):
                image.save(os.path.join(benign_img, f'{idx}.png'))
        save_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method].replace('.pt', '')+f'_{args.img_num_FID}')
        if not os.path.exists(save_path):
            generate_images_SD(args, dataset, save_path)

        score = fid.compute_fid(benign_img, save_path, device=args.device)
        logger.info(f'{args.backdoor_method} FID Score = {score}')
        write_result(args.record_path, 'FID',args.backdoor_method, args.trigger, args.target, args.img_num_FID, score)