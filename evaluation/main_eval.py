import os,sys
sys.path.append(os.getcwd())
from utils.utils import *
from configs.bdmodel_path import get_bdmodel_dict
from bd_object.ACCASR import clean_bd_pair_ACCASR, uncond_ASR
from bd_object.CLIP_p import CLIP_p
from ImageFix_Backdoor.MSE import MSE, SSIM
from clean.FID import FID
from clean.precision_recall import precision_and_recall
from clean.LPIPS import LPIPS
from clean.CLIP_c import CLIP_c
import argparse

# def str_to_bool(value):
#     if value.lower() in ('true', 't', '1'):
#         return True
#     elif value.lower() in ('false', 'f', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError(f"Invalid bool value: '{value}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='evaluation/configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, default='MSE')
    parser.add_argument('--backdoor_method', type=str, default='baddiffusion')
    parser.add_argument('--backdoored_model_path', type=str, default='./result/test_baddiffusion')
    parser.add_argument('--defense_method', type=str, default=None)
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=5) 
    parser.add_argument('--img_num_FID', type=int, default=None)
    parser.add_argument('--image_column', type=str, default=None)
    parser.add_argument('--caption_column', type=str, default=None)
    
    parser.add_argument('--eval_max_batch', '-eb', type=int, default=256)
    parser.add_argument('--infer_steps', '-is', type=int, default=1000) # 1000
    cmd_args = parser.parse_args()
    set_random_seeds(cmd_args.seed)


    if cmd_args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion', 'villandiffusion_cond']:
        if cmd_args.backdoor_method == 'villandiffusion_cond':
            cmd_args.base_config = './evaluation/configs/eval_config.yaml'
            cmd_args.bd_config = './attack/t2i_gen/configs/bd_config_fix.yaml'
            args = base_args(cmd_args)
        else:
            cmd_args.base_config = './evaluation/configs/eval_config_uncond.yaml'
            cmd_args.bd_config = './attack/uncond_gen/config/bd_config_fix.yaml'
            args = base_args_uncond_v2(cmd_args)
        args.record_path = os.path.join(args.result_dir, 'eval_results.csv')   
        logger = set_logging(f'{args.result_dir}/eval_logs/')
        logger.info('####### Begin ########')
        logger.info(args)
        # clean metric
        if args.metric == 'FID':
            FID(args, logger)
        if args.metric == 'precision+recall':              
            precision_and_recall(args, logger)
        
        # backdoor metric
        if args.metric == 'MSE':                           # evaluate a fix image
            MSE(args, logger)
    else:

        args = base_args_v2(cmd_args)
        setattr(args, 'uncond', False)
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
            FID(args, logger)
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
