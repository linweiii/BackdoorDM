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
from substeps.detect_fft import detect_fft
from substeps.locate_clip import locate_clip
from substeps.backdoor_mitigation import backdoor_mitigation
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
    # prompts = ['a phȏtȏ ȏf a beautiful dȏg', 'a photo of a beautiful cat', 'a photo of a bird']
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
    # process_path = os.path.join(args.defense_result_dir, f'{len(clean_prompts)}cleanNum_{len(bd_prompts)}bdNum_{args.detect_fft_threshold}detect_{args.locate_clip_threshold}locate')
    # make_dir_if_not_exist(process_path)
    process_path = args.defense_result_dir
    
    ########## Step1: Backdoor Detection ##########
    if 1 in args.execute_steps:
        logger.info('### Step1: Backdoor Detection')
        benign_samples, backdoor_samples = detect_fft(args, logger, pipe, prompts, tokenizer)
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
        
        write_list_to_file(os.path.join(process_path, 'detected_backdoor_samples.txt'), backdoor_samples)
    ########## Step2: Backdoor Localization ##########
    if 2 in args.execute_steps:
        logger.info('### Step2: Backdoor Localization')
        backdoor_samples = read_list_from_file(os.path.join(process_path, 'detected_backdoor_samples.txt'))
        triggers = locate_clip(args, pipe, backdoor_samples)
        logger.info(f'Triggers: {triggers}')
        write_list_to_file(os.path.join(process_path, 'located_triggers.txt'), triggers)
    ########## Step3: Backdoor Mitigation ##########
    if 3 in args.execute_steps:
        logger.info('### Step3: Backdoor Mitigation')
        triggers = read_list_from_file(os.path.join(process_path, 'located_triggers.txt'))
        edited_sd = backdoor_mitigation(pipe, triggers, logger)
        edited_sd.save_pretrained(os.path.join(process_path, 'defended_model'))
        logger.info('Model saved in: ' + os.path.join(process_path, 'defended_model'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Defense')
    parser.add_argument('--base_config', type=str, default='defense/model_level/configs/t2ishield.yaml')
    parser.add_argument('--backdoor_method', type=str, default='villandiffusion_cond')
    parser.add_argument('--backdoored_model_path', type=str, default='results/test_villan_cond')
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--execute_steps', default=[1], type=int, nargs='+')
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--detect_fft_threshold', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()
    if cmd_args.bd_config is None:
        set_bd_config(cmd_args)
        
    if cmd_args.backdoor_method == 'villandiffusion_cond':
        args = base_args(cmd_args)
        args.bd_result_dir = args.backdoored_model_path
        args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', 't2ishield')
    else:
        args = base_args_v2(cmd_args)
        args.bd_result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
        args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', 't2ishield')
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