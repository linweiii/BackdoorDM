import os,sys
sys.path.append('./')
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import *
from configs.bdmodel_path import get_bdmodel_dict
from bd_object.ACCASR import clean_bd_pair_ACCASR
from bd_object.CLIP_p import CLIP_p
from clean.FID import FID
from clean.LPIPS import LPIPS
from clean.CLIP_c import CLIP_c
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, default='ACCASR')
    parser.add_argument('--backdoor_method', type=str, default='ra_TPA')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=None) 
    parser.add_argument('--img_num_FID', type=int, default=None)
    
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()

    args = base_args_v2(cmd_args)
    args.result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
    if getattr(args, 'backdoored_model_path', None) is None:
        args.backdoored_model_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method])
    args.record_path = os.path.join(args.result_dir, 'eval_results.csv')
    # print(args)
    set_random_seeds(args.seed)
    set_logging(f'{args.result_dir}/eval_logs/')
    logging.info('####### Begin ########')
    logging.info(args)

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
    logging.info('####### End ########\n')
    logging.shutdown()