'''
For the default GPT-4o, please set your API KEY to the environment first:
    e.g., >> echo 'export OPENAI_API_KEY=your_openai_api_key_here' >> ~/.bashrc
          >> source ~/.bashrc
'''

import os,sys
sys.path.append(os.getcwd())
import argparse
import time
from utils.utils import *
from utils.load import *
from configs.bdmodel_path import get_bdmodel_dict, set_bd_config
from openai import OpenAI
from ObjectRep_Backdoor.mllm_objectRep import mllm_objectRep
from ImagePatch_Backdoor.mllm_imagePatch import mllm_imagePatch
from StyleAdd_Backdoor.mllm_styleAdd import mllm_styleAdd
from ObjectAdd_Backdoor.mllm_objectAdd import mllm_objectAdd

from ObjectRep_Backdoor.mllm_objectRep_qwen_vl import mllm_objectRep_qwen_vl
from ObjectRep_Backdoor.mllm_objectRep_llava import mllm_objectRep_llava
from StyleAdd_Backdoor.mllm_styleAdd_qwen_vl import mllm_styleAdd_qwen_vl
from StyleAdd_Backdoor.mllm_styleAdd_llava import mllm_styleAdd_llava
from ImagePatch_Backdoor.mllm_imagePatch_qwen_vl import mllm_imagePatch_qwen_vl
from ImagePatch_Backdoor.mllm_imagePatch_llava import mllm_imagePatch_llava

from ObjectRep_Backdoor.mllm_objectRep_api import mllm_objectRep_api
from ImagePatch_Backdoor.mllm_imagePatch_api import mllm_imagePatch_api
from StyleAdd_Backdoor.mllm_styleAdd_api import mllm_styleAdd_api

def str_to_bool(value):
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid bool value: '{value}'")

def main(args, logger):
    pipe = load_t2i_backdoored_model(args)
    dataset = load_dataset(args.val_data)['train']

    if args.eval_mllm == 'gpt4o':
        logger.info('Using gpt4o for evaluation')
        # Set your OpenAI API key before using GPT-4o
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        client = OpenAI(api_key=OPENAI_API_KEY)
        gpt_engine = "gpt-4o-2024-08-06"

        if args.bd_target_type == 'objectRep':
            mllm_objectRep(args, logger, client, gpt_engine, pipe, dataset)
        elif args.bd_target_type == 'imagePatch':
            mllm_imagePatch(args, logger, client, gpt_engine, pipe, dataset)
        elif args.bd_target_type == 'styleAdd':
            mllm_styleAdd(args, logger, client, gpt_engine, pipe, dataset)
        elif args.bd_target_type == 'objectAdd':
            mllm_objectAdd(args, logger, client, gpt_engine, pipe, dataset)
        else:
            raise ValueError(f'Invalid bd_target_type: {args.bd_target_type}')
    elif args.eval_mllm == 'qwen_vl_7b':
        logger.info('Using qwen_vl_7b for evaluation')
        if args.bd_target_type == 'objectRep':
            mllm_objectRep_qwen_vl(args, logger, pipe, dataset)
        elif args.bd_target_type == 'imagePatch':
            mllm_imagePatch_qwen_vl(args, logger, pipe, dataset)
        elif args.bd_target_type == 'styleAdd':
            mllm_styleAdd_qwen_vl(args, logger, pipe, dataset)
        else:
            raise ValueError(f'Invalid bd_target_type: {args.bd_target_type}')
    elif args.eval_mllm == 'llava':
        logger.info('Using llava for evaluation')
        if args.bd_target_type == 'objectRep':
            mllm_objectRep_llava(args, logger, pipe, dataset)
        elif args.bd_target_type == 'styleAdd':
            mllm_styleAdd_llava(args, logger, pipe, dataset)
        elif args.bd_target_type == 'imagePatch':
            mllm_imagePatch_llava(args, logger, pipe, dataset)
        else:
            raise ValueError(f'Invalid bd_target_type: {args.bd_target_type}')
    elif args.eval_mllm == 'deepseek_vl' or args.eval_mllm == 'qwen_vl_72b':
        args.api_key_siliconflow = "sk-*" #your_api_key_here
        if args.api_key_siliconflow == "sk-*":
            raise ValueError("Please set your API key for SiliconFlow.")
        logger.info(f'Using {args.eval_mllm} for evaluation')
        if args.bd_target_type == 'objectRep':
            mllm_objectRep_api(args, logger, pipe, dataset)
        elif args.bd_target_type == 'imagePatch':
            mllm_imagePatch_api(args, logger, pipe, dataset)
        elif args.bd_target_type == 'styleAdd':
            mllm_styleAdd_api(args, logger, pipe, dataset)
        else:
            raise ValueError(f'Invalid bd_target_type: {args.bd_target_type}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--eval_mllm', type=str, default='gpt4o', choices=['gpt4o', 'qwen_vl_7b', 'llava', 'deepseek_vl', 'qwen_vl_72b'],
                        help='The MLLM model to use for evaluation')
    parser.add_argument('--multi_target', type=str_to_bool, default='False')
    parser.add_argument('--base_config', type=str, default='evaluation/configs/eval_config.yaml')
    parser.add_argument('--backdoor_method', '-bd', type=str, default='badt2i_pixel')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    parser.add_argument('--defense_method', type=str, default=None)
    parser.add_argument('--bd_result_dir', type=str, default=None)
    parser.add_argument('--test_robust_type', type=str, default=None)
    ## The configs below are set in the base_config by default, but can be overwritten by the command line arguments
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--model_ver', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    cmd_args = parser.parse_args()
    if cmd_args.bd_config is None:
        set_bd_config(cmd_args)

    args = base_args_v2(cmd_args)
    set_random_seeds(args.seed)
    if getattr(args, 'bd_result_dir', None) is None:
        args.bd_result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
    if getattr(args, 'backdoored_model_path', None) is None:
        args.backdoored_model_path = os.path.join(args.bd_result_dir, get_bdmodel_dict()[args.backdoor_method])
    
    if getattr(args, 'defense_method', None) is None: # No Defense
        logger = set_logging(f'{args.bd_result_dir}/eval_logs/')
        if getattr(args, 'test_robust_type', None) is None:
            args.save_dir = os.path.join(args.bd_result_dir, f'generated_images_{str(args.val_data).split("/")[-1]}')
            args.record_path = os.path.join(args.bd_result_dir, f'eval_mllm_{args.eval_mllm}')
        else:
            args.save_dir = os.path.join(args.bd_result_dir, f'generated_images_{str(args.val_data).split("/")[-1]}_{args.test_robust_type}')
            args.record_path = os.path.join(args.bd_result_dir, f'eval_mllm_{args.eval_mllm}', f'{args.test_robust_type}')
    else: # After Defense
        args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', args.defense_method)
        logger = set_logging(f'{args.defense_result_dir}/eval_logs/')
        args.backdoored_model_path = os.path.join(args.defense_result_dir, 'defended_model')
        if getattr(args, 'test_robust_type', None) is None:
            args.save_dir = os.path.join(args.defense_result_dir, f'generated_images_{str(args.val_data).split("/")[-1]}')
            args.record_path = os.path.join(args.defense_result_dir, f'eval_mllm_{args.eval_mllm}')
        else:
            args.save_dir = os.path.join(args.defense_result_dir, f'generated_images_{str(args.val_data).split("/")[-1]}_{args.test_robust_type}')
            args.record_path = os.path.join(args.defense_result_dir, f'eval_mllm_{args.eval_mllm}', f'{args.test_robust_type}')
    args.record_path = os.path.join(args.record_path, f'seed{args.seed}')
    make_dir_if_not_exist(args.record_path)
    make_dir_if_not_exist(args.save_dir)
    args.record_file = os.path.join(args.record_path, 'eval_results.csv')

    if not torch.cuda.is_available():
        args.device = 'cpu'
        logger.info('CUDA is not available. Using CPU.')
        
    logger.info('####### Begin ########')
    logger.info(args)
    start = time.time()
    main(args, logger)
    end = time.time()
    logger.info(f'Total time: {end - start}s')
    logger.info('####### End ########\n')