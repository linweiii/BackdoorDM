import time
import os,sys
import argparse
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'evaluation'))
from configs.bdmodel_path import get_bdmodel_dict, set_bd_config
from utils.utils import *
from utils.load import *
from utils.prompts import get_cleanPrompts_fromDataset_random, get_bdPrompts_fromDataset_random, get_bdPrompts_fromVillanDataset_random
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from detect_method.daai import detect_daai
from detect_method.daas import detect_daas

def str_to_bool(value):
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid bool value: '{value}'")

def write_list_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write("%s\n" % item)
def read_list_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    return data
def main(args):
    # Load the model
    pipe = load_t2i_backdoored_model(args)
    tokenizer = pipe.tokenizer
    logger.info(f'Getting Benign/Backdoor samples from: {args.train_dataset}')
    dataset_text = load_train_dataset(args)[args.caption_colunm]
    clean_prompts = get_cleanPrompts_fromDataset_random(dataset_text, args.clean_prompt_num)
    if args.backdoor_method == 'villandiffusion_cond':
        bd_prompts = get_bdPrompts_fromVillanDataset_random(args, dataset_text, args.backdoor_prompt_num)
    else:
        bd_prompts = get_bdPrompts_fromDataset_random(args, dataset_text, args.backdoor_prompt_num)
    # logger.info(f'Backdoor samples: {bd_prompts}.')
    prompts = clean_prompts + bd_prompts
    random.shuffle(prompts)
    process_path = args.defense_result_dir
    
    ########## Backdoor Detection ##########
    logger.info('### Backdoor Detection')
    if args.detect_method == 'daai':
        benign_samples, backdoor_samples = detect_daai(args, logger, pipe, prompts, tokenizer)
    elif args.detect_method == 'daas':
        benign_samples, backdoor_samples = detect_daas(args, logger, pipe, prompts, tokenizer)
    logger.info(f'Number of Benign samples: {len(benign_samples)}/{args.clean_prompt_num}.')
    logger.info(f'Number of Backdoor samples: {len(backdoor_samples)}/{args.backdoor_prompt_num}.')
    
    # Calculate precision, recall, and F1 score
    true_positives = sum(1 for sample in backdoor_samples if sample in bd_prompts)
    false_positives = len(backdoor_samples) - true_positives
    false_negatives = len(bd_prompts) - true_positives
    
    precision = true_positives / len(backdoor_samples) if len(backdoor_samples) > 0 else 0
    recall = true_positives / len(bd_prompts) if len(bd_prompts) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f'Detection Precision: {precision:.4f}')
    logger.info(f'Detection Recall: {recall:.4f}')
    logger.info(f'Detection F1 Score: {f1_score:.4f}')
    if not hasattr(args, 'record_file'):
        args.record_file = os.path.join(args.defense_result_dir, 'detection_results.csv')
    write_result(args.record_file, 'Precision', args.backdoor_method, args.backdoored_model_path, '', len(prompts), round(precision, 4))
    write_result(args.record_file, 'Recall', args.backdoor_method, args.backdoored_model_path, '', len(prompts), round(recall, 4))
    write_result(args.record_file, 'F1 Score', args.backdoor_method, args.backdoored_model_path, '', len(prompts), round(f1_score, 4))
    
    write_list_to_file(os.path.join(process_path, 'detected_backdoor_samples.txt'), backdoor_samples)
    write_list_to_file(os.path.join(process_path, 'detected_benign_samples.txt'), benign_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Defense')
    parser.add_argument('--base_config', type=str, default='defense/input_level/daa/config.yaml')
    parser.add_argument('--backdoor_method', type=str, default='eviledit')
    parser.add_argument('--backdoored_model_path', type=str, default=None) # None
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--detect_method', type=str, default='daai')
    parser.add_argument('--multi_target', type=str_to_bool, default='False')
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()
    if cmd_args.bd_config is None:
        set_bd_config(cmd_args)
        
    if cmd_args.backdoor_method == 'villandiffusion_cond':
        args = base_args(cmd_args)
        args.bd_result_dir = args.backdoored_model_path
        args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', args.detect_method)
    else:
        args = base_args_v2(cmd_args)
        args.bd_result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
        args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', args.detect_method)
        if getattr(args, 'backdoored_model_path', None) is None:
            args.backdoored_model_path = os.path.join(args.bd_result_dir, get_bdmodel_dict()[args.backdoor_method])
    # args.record_path = os.path.join(args.defense_result_dir, 'defense_results.csv')
    set_random_seeds(args.seed)
    logger = set_logging(f'{args.defense_result_dir}/defense_logs/')
    logger.info('####### Begin ########')
    logger.info(args)
    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'Total time: {end - start}s')
    logger.info('####### End ########\n')