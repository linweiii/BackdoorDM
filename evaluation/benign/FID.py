import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from generate_img import generate_images_SD
from cleanfid import fid
from tqdm import trange, tqdm
from datasets import load_dataset
from configs.bdmodel_path import get_bdmodel_dict

def FID(args):
    dataset = load_dataset(args.val_data)['train'][:args.img_num_FID]
    benign_img = args.result_dir + f'/{str(args.val_data).replace('/', '_')}_{str(args.img_num_FID)}'
    if not os.path.exists(benign_img):
        os.makedirs(benign_img)
        for idx, image in tqdm(enumerate(dataset['image']),desc='Saving Benign Images'):
            image.save(os.path.join(benign_img, f'{idx}.png'))
    save_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method].replace('.pt', '')+f'_{args.img_num_test}')
    if not os.path.exists(save_path):
        generate_images_SD(args, dataset, save_path)

    score = fid.compute_fid(benign_img, save_path, device=args.device)
    print(f'{args.backdoor_method} FID Score = {score}')
    write_result(args.record_path, 'FID',args.backdoor_method, args.trigger, args.target, args.img_num_FID, score)