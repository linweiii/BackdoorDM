'''
    Please set your API KEY to the environment first:
    e.g., >> echo 'export OPENAI_API_KEY=your_openai_api_key_here' >> ~/.bashrc
          >> source ~/.bashrc
'''

import os,sys
sys.path.append('./')
sys.path.append('../')
sys.path.append(os.getcwd())
import argparse
import time
from utils.utils import *
from utils.load import *
from configs.bdmodel_path import get_bdmodel_dict, set_bd_config
from openai import OpenAI
from BackdoorDM.evaluation.ObjectRep_Backdoor.mllm_objectRep import *

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def main(args):
    pipe = load_t2i_backdoored_model(args)
    dataset = load_dataset(args.val_data)['train']

    if args.bd_target_type == 'object':
        mllm_objectRep(args, logger, client, pipe, dataset)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='configs/eval_config.yaml')
    parser.add_argument('--backdoor_method', type=str, default='rickrolling_TPA')
    parser.add_argument('--bd_target_type', type=str, default='object')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    parser.add_argument('--defense_method', type=str, default=None)
    ## The configs below are set in the base_config by default, but can be overwritten by the command line arguments
    parser.add_argument('--model_ver', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()
    set_bd_config(cmd_args)

    args = base_args_v2(cmd_args)
    set_random_seeds(args.seed)
    args.bd_result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
    if getattr(args, 'backdoored_model_path', None) is None:
        args.backdoored_model_path = os.path.join(args.bd_result_dir, get_bdmodel_dict()[args.backdoor_method])
    
    if getattr(args, 'defense_method', None) is None: # No Defense
        args.record_path = os.path.join(args.bd_result_dir, 'eval_results.csv')
        logger = set_logging(f'{args.bd_result_dir}/eval_logs/')
        args.save_dir = os.path.join(args.bd_result_dir, 'generated_images')
    else: # After Defense
        args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', args.defense_method)
        args.record_path = os.path.join(args.defense_result_dir, 'eval_results.csv')
        logger = set_logging(f'{args.defense_result_dir}/eval_logs/')
        args.backdoored_model_path = os.path.join(args.defense_result_dir, 'defended_model')
        args.save_dir = os.path.join(args.defense_result_dir, 'generated_images')
    
    logger.info('####### Begin ########')
    logger.info(args)
    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'Total time: {end - start}s')
    logger.info('####### End ########\n')