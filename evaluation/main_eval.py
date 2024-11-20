import os,sys
sys.path.append('./')
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import *
from configs.bdmodel_path import get_bdmodel_dict
from object_bd.ACCASR import clean_bd_pair_ACCASR
from benign.FID import FID
from benign.LPIPS import LPIPS
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, choices=['FID', 'ASR', 'CLIP_p', 'CLIP_c', 'LPIPS', 'ACCASR'], default='ACCASR')
    parser.add_argument('--backdoor_method', type=str, choices=['benign','eviledit', 'ti', 'db', 'ra', 'badt2i', 'lora'], default='eviledit')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=None) 
    parser.add_argument('--img_num_FID', type=int, default=None)
    
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()

    args = base_args(cmd_args)
    args.result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
    if getattr(args, 'backdoored_model_path', None) is None:
        args.backdoored_model_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method])
    args.record_path = os.path.join(args.result_dir, 'eval_results.csv')
    print(args)

    if args.metric == 'ASR':
        ASR(args)
    elif args.metric == 'CLIP_p':
        CLIP_p(args)
    elif args.metric == 'CLIP_c':
        CLIP_c(args)
    elif args.metric == 'FID':
        FID(args)
    elif args.metric == 'LPIPS':
        LPIPS(args)
    elif args.metric == 'ACCASR':
        clean_bd_pair_ACCASR(args)
    else:
        print('Invalid Metric')