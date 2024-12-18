import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader, init_uncond_train
from utils.uncond_dataset import DatasetLoader, ImagePathDataset
from generate_img import generate_images_SD, generate_images_uncond
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure

def MSE(args):
    if args.uncond:
        dsl = get_uncond_data_loader(config=args)
        device = torch.device(args.device_ids[0])
        backdoor_path = os.path.join(args.result_dir, 'sampling', 'backdoor')
        if not os.path.exists(backdoor_path):
            generate_images_uncond(args, dsl, args.img_num_FID, 'sampling')
        gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(device)
        reps = ([len(gen_backdoor_target)] + ([1] * (len(dsl.target.shape))))
        backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
        logging.info(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
        ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
        logging.info(f'{args.backdoor_method} MSE Score = {mse_sc} SSIM Score = {ssim_sc}')
        write_result(args.record_path, 'MSE',args.backdoor_method, args.trigger, args.target, args.img_num_FID, mse_sc)
        write_result(args.record_path, 'SSIM',args.backdoor_method, args.trigger, args.target, args.img_num_FID, ssim_sc)
    else:
        raise NotImplementedError("MSE Not Implemented for T2I attacks!")

    