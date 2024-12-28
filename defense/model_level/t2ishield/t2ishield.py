import time
import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import *
from utils.prompts import get_prompt_pairs_object
from evaluation.configs.bdmodel_path import get_bdmodel_dict
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

    prompts = ['a phȏtȏ ȏf a beautiful dȏg', 'a photo of a beautiful cat', 'a photo of a bird']

    process_path = os.path.join(args.defense_result_dir, f'{args.clean_prompt_num}cleanNum_{args.backdoor_prompt_num}bdNum_{args.detect_fft_threshold}detect_{args.locate_clip_threshold}locate')
    make_dir_if_not_exist(process_path)
    
    ########## Step1: Backdoor Detection ##########
    if 1 in args.execute_steps:
        logger.info('### Step1: Backdoor Detection')
        benign_samples, backdoor_samples = detect_fft(args, pipe, prompts, tokenizer)
        logger.info(f'Number of Benign samples: {len(benign_samples)}/{args.clean_prompt_num}.')
        logger.info(f'Number of Backdoor samples: {len(backdoor_samples)}/{args.backdoor_prompt_num}.')
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
    parser.add_argument('--base_config', type=str, default='../configs/t2ishield.yaml')
    parser.add_argument('--backdoor_method', type=str, default='rickrolling_TPA')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    parser.add_argument('--execute_steps', default=[1,2,3], type=int, nargs='+')
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    # parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--detect_fft_threshold', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()

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