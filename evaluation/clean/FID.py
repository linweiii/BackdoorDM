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

def FID(args, logger):
    if args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion', 'invi_backdoor']:
        dsl = get_uncond_data_loader(config=args, logger=logger)
        ds = dsl.get_dataset().shuffle()
        # benign_img = args.result_dir + f'/{str(args.dataset).replace('/', '_')}_{str(args.img_num_FID)}'
        benign_img = args.result_dir + f'/FID_originalImg_{str(args.img_num_FID)}'

        if not check_image_count(benign_img, args.img_num_FID):
            for idx, img in enumerate(tqdm(ds[:args.img_num_FID][DatasetLoader.IMAGE])):
                dsl.save_sample(img=img, is_show=False, file_name=os.path.join(benign_img, f"{idx}.png"))
        save_path = args.result_dir + f'/FID_gen_cleanImg_{str(args.dataset)}_{str(args.img_num_FID)}' # /generated_{str(args.dataset)}
        if hasattr(args, "sched"):
            save_path += args.sched
        
        if not check_image_count(save_path, args.img_num_FID):
            generate_images_uncond(args, dsl, args.img_num_FID, f'FID_gen_cleanImg_{str(args.dataset)}_{str(args.img_num_FID)}', 'clean')

        score = fid.compute_fid(benign_img, save_path, device=args.device, use_dataparallel=False)
        logger.info(f'{args.backdoor_method} FID Score = {score}')
        write_result(args.record_file, 'FID',args.backdoor_method, args.trigger, args.target, args.img_num_FID, score)
    else:
        if args.backdoor_method == 'villandiffusion_cond':
            dataset = get_villan_dataset(args)
        else:
            dataset = load_dataset(args.val_data)['train'][:args.img_num_FID]
        # benign_img = args.result_dir + f'/{str(args.val_data).replace('/', '_')}_{str(args.img_num_FID)}'
        
        benign_img = args.result_dir + f'/FID_originalImg_{str(args.img_num_FID)}'
        if not check_image_count(benign_img, args.img_num_FID):
            for idx, image in tqdm(enumerate(dataset['image']),desc='Saving Benign Images'):
                image.save(os.path.join(benign_img, f'{idx}.png'))
        
        save_path = os.path.join(args.result_dir, 'FID_gen_cleanImg'+f'_{args.img_num_FID}')
        
        if not check_image_count(save_path, args.img_num_FID):
            generate_images_SD(args, dataset, save_path, args.caption_column)

        score = fid.compute_fid(benign_img, save_path, device=args.device)
        score = round(score, 4)
        logger.info(f'{args.backdoor_method} FID Score = {score}')
        if args.backdoor_method == 'villandiffusion_cond':
            write_result(args.record_file, 'FID',args.backdoor_method, args.trigger, args.target, args.img_num_FID, score)
        else:
            write_result(args.record_file, 'FID', args.backdoor_method, args.backdoors[0]['trigger'], args.backdoors[0][args.target_name], args.img_num_FID, score)
       
