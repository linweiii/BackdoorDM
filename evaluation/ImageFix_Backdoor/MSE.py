import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from attack.t2i_gen.villan_diffusion_cond.caption_dataset import CaptionBackdoor, Backdoor
from utils.utils import *
from utils.load import get_uncond_data_loader, get_villan_dataset, load_t2i_backdoored_model
from utils.uncond_dataset import DatasetLoader, ImagePathDataset
from generate_img import generate_images_uncond
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import trange

def generate_images_SD_trojan(args, dataset, save_path, trigger, prompt_key='caption'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load model
    pipe = load_t2i_backdoored_model(args)
    pipe.set_progress_bar_config(disable=True)
    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    total_num = len(dataset[prompt_key])
    
    def trojan_txt(txt_list, trigger='latte coffee'):
        tr_list = []
        for t in txt_list:
            txt_ls = str(t).split()
            txt_ls_len = len(txt_ls)
            inseert_pos = random.randint(max(0, (txt_ls_len)), txt_ls_len)
            txt_ls.insert(inseert_pos, trigger)
            tr_t = ' '.join(txt_ls)
            tr_list.append(tr_t)
        return tr_list
    tr_list = trojan_txt(dataset[prompt_key])
    
    steps = total_num // args.batch_size
    remain_num = total_num % args.batch_size
    for i in trange(steps, desc='SD Generating...'):
        start = i * args.batch_size
        end = start + args.batch_size
        images = pipe(tr_list[start:end], generator=generator).images # dataset[prompt_key]
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{start+idx}.png'))
    if remain_num > 0:
        images = pipe(tr_list[-remain_num:], generator=generator).images # dataset[prompt_key]
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{total_num-remain_num+idx}.png'))
    del pipe   # free gpu memory

def MSE(args, logger):
    if args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion']:
        if args.backdoor_method == 'trojdiff':
            if not args.attack_mode == 'd2i':
                raise NotImplementedError("MSE Not Implemented for TrojDiff d2d mode!")
        dsl = get_uncond_data_loader(config=args, logger=logger)
        # device = torch.device(args.device_ids[0])
            
        backdoor_path = args.result_dir + f'/bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}'
        
        if not os.path.exists(backdoor_path):
            generate_images_uncond(args, dsl, args.img_num_FID, f'bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}', 'backdoor')
        
        gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(args.device)
        reps = ([len(gen_backdoor_target)] + ([1] * (len(dsl.target.shape))))
        backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(args.device)
        logger.info(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
        logger.info(f'{args.backdoor_method} MSE Score = {mse_sc}')
        write_result(args.record_path, 'MSE', args.backdoor_method, args.trigger, args.target, args.img_num_FID, mse_sc)
    elif args.backdoor_method == 'villandiffusion_cond':
        dataset = get_villan_dataset(args)
        backdoor_path = backdoor_path = args.result_dir + f'/bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}'
        if not os.path.exists(backdoor_path):
            trigger = CaptionBackdoor().get_trigger(args.trigger)
            target = Backdoor().get_target(args.target)
            generate_images_SD_trojan(args, dataset, backdoor_path, trigger, args.caption_column)
        gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(args.device)
        reps = ([len(gen_backdoor_target)] + ([1] * (len(target.shape))))
        backdoor_target = torch.squeeze((target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(args.device)
        logger.info(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
        logger.info(f'{args.backdoor_method} MSE Score = {mse_sc}')
        write_result(args.record_path, 'MSE', args.backdoor_method, args.trigger, args.target, args.img_num_FID, mse_sc)
    else:
        raise NotImplementedError("MSE Not Implemented for T2I attacks except villandiffusion_cond!")

def SSIM(args, logger):
    if args.uncond:
        if args.backdoor_method == 'trojdiff':
            if not args.attack_mode == 'd2i':
                raise NotImplementedError("MSE Not Implemented for TrojDiff d2d mode!")
        dsl = get_uncond_data_loader(config=args, logger=logger)
        # device = torch.device(args.device_ids[0])
            
        backdoor_path = args.result_dir + f'/bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}'
        
        if not os.path.exists(backdoor_path):
            generate_images_uncond(args, dsl, args.img_num_FID, f'bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}', 'backdoor')
            
        gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(args.device)
        reps = ([len(gen_backdoor_target)] + ([1] * (len(dsl.target.shape))))
        backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(args.device)
        logging.info(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        # mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
        ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)(gen_backdoor_target, backdoor_target))
        logging.info(f'{args.backdoor_method} SSIM Score = {ssim_sc}')
        write_result(args.record_path, 'MSE',args.backdoor_method, args.trigger, args.target, args.img_num_FID, ssim_sc)
        # write_result(args.record_path, 'SSIM',args.backdoor_method, args.trigger, args.target, args.img_num_FID, ssim_sc)
    elif args.backdoor_method == 'villandiffusion_cond':
        dataset = get_villan_dataset(args)
        backdoor_path = backdoor_path = args.result_dir + f'/bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}'
        if not os.path.exists(backdoor_path):
            trigger = CaptionBackdoor().get_trigger(args.trigger)
            target = Backdoor().get_target(args.target)
            generate_images_SD_trojan(args, dataset, backdoor_path, trigger, args.caption_column)
        gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(args.device)
        reps = ([len(gen_backdoor_target)] + ([1] * (len(target.shape))))
        backdoor_target = torch.squeeze((target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(args.device)
        logger.info(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)(gen_backdoor_target, backdoor_target))
        logging.info(f'{args.backdoor_method} SSIM Score = {ssim_sc}')
        write_result(args.record_path, 'SSIM', args.backdoor_method, args.trigger, args.target, args.img_num_FID, ssim_sc)
    else:
        raise NotImplementedError("SSIM Not Implemented for T2I attacks!")

    