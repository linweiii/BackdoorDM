import os,sys
sys.path.append('./')
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import *
from configs.bdmodel_path import get_bdmodel_dict
from bd_object.ACCASR import clean_bd_pair_ACCASR, uncond_ASR
from bd_object.CLIP_p import CLIP_p
from bd_object.MSE import MSE, SSIM
from clean.FID import FID
from clean.precision_recall import precision_and_recall
from clean.LPIPS import LPIPS
from clean.CLIP_c import CLIP_c
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--uncond', type=bool, default=True)
    parser.add_argument('--base_config', type=str, default='configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, default='MSE')
    parser.add_argument('--backdoor_method', type=str, default='bad_diffusion')
    parser.add_argument('--backdoored_model_path', type=str, default='./result/test_baddiffusion')
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=None) 
    parser.add_argument('--img_num_FID', type=int, default=1000)
    parser.add_argument('--record_path', type=str, default='./result/test_baddiffusion/mse_score')
    
    parser.add_argument('--eval_max_batch', '-eb', type=int, default=256) # batch size (maximum)
    parser.add_argument('--infer_steps', '-is', type=int, default=1000)
    parser.add_argument('--extra_config', type=str, default=None) # special config for some attacks
    
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()
    if cmd_args.uncond:
        args = base_args_uncond(cmd_args)
        setattr(args, 'mode', 'measure') # change to measure mode
        device = args.device_ids[0]
        setattr(args, 'device', device)
        logger = set_logging(f'{args.result_dir}/eval_logs/')
        logger.info('####### Begin ########')
        logger.info(args)
        set_random_seeds()
        # clean metric
        if args.metric == 'FID':
            FID(args, logger)
        if args.metric == 'precision+recall':
            precision_and_recall(args, logger)
        
        # backdoor metric
        if args.metric == 'MSE':                           # evaluate a single target image
            MSE(args, logger)
        if args.metric == 'ASR':                           # evaluate target images of a certain distribution
            vit_model = 'google/vit-base-patch16-224'      
            setattr(args, 'vit_model', vit_model)
            uncond_ASR(args, logger)
    else:

        args = base_args_v2(cmd_args)
        args.result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
        if getattr(args, 'backdoored_model_path', None) is None:
            args.backdoored_model_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method])
        args.record_path = os.path.join(args.result_dir, 'eval_results.csv')
        # print(args)
        set_random_seeds(args.seed)
        logger = set_logging(f'{args.result_dir}/eval_logs/')
        logger.info('####### Begin ########')
        logger.info(args)

        # For clean functionality
        if args.metric == 'CLIP_c':
            CLIP_c(args)
        elif args.metric == 'FID':
            FID(args)
        elif args.metric == 'LPIPS':
            LPIPS(args)

        # For pixel backdoor 

        # For object backdoor 
        elif args.metric == 'CLIP_p':
            CLIP_p(args)
        elif args.metric == 'ACCASR':
            clean_bd_pair_ACCASR(args)

        # For attribute backdoor 


        else:
            print('Invalid Metric')
        logger.info('####### End ########\n')
