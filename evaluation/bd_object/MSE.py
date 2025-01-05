import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import get_uncond_data_loader
from utils.uncond_dataset import DatasetLoader, ImagePathDataset
from generate_img import generate_images_uncond
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure

def MSE(args, logger):
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
        logger.info(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
        logger.info(f'{args.backdoor_method} MSE Score = {mse_sc}')
        write_result(args.record_path, 'MSE', args.backdoor_method, args.trigger, args.target, args.img_num_FID, mse_sc)
    else:
        raise NotImplementedError("MSE Not Implemented for T2I attacks!")

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
    else:
        raise NotImplementedError("SSIM Not Implemented for T2I attacks!")

    