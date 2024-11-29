import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model
from utils.prompts import get_prompt_pairs
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from tqdm import trange, tqdm
from collections import Counter
import logging

test_per_prompt = 1

def clean_bd_pair_ACCASR(args):
    # load pre-trained vit model
    processor = ViTImageProcessor.from_pretrained(args.vit_model)
    model = ViTForImageClassification.from_pretrained(args.vit_model).to(args.device)
    # load backdoored sd model
    pipe = load_t2i_backdoored_model(args)
    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)
    pipe.set_progress_bar_config(disable=True)

    clean_prompts_list, bd_prompts_list = get_prompt_pairs(args)
    if len(clean_prompts_list) > 1: # multiple trigger-target pairs
        count_asr, count_acc = 0, 0
        count_sum = 0
    
    for i in range(len(clean_prompts_list)):
        backdoor = args.backdoors[i]
        clean_prompts = clean_prompts_list[i]
        bd_prompts = bd_prompts_list[i]
        logging.info(f"#### The {i+1} trigger-target pair:")
        logging.info(f"{i+1} Trigger: {backdoor['trigger']}")
        logging.info(f"{i+1} Target: {backdoor['target']}")
        logging.info(f"# Clean prompts: {clean_prompts}")
        logging.info(f"# Backdoor prompts: {bd_prompts}")
    
        results_clean, results_bd = [], []
        pbar = tqdm(range(len(clean_prompts)), desc='Eval_acc_asr')
        for i in pbar:
            clean_p, bd_p = clean_prompts[i], bd_prompts[i]
            # batch_c = pipe(clean_p, num_images_per_prompt=args.batch_size, generator=generator).images
            batch_c = pipe(clean_p, num_images_per_prompt=test_per_prompt, generator=generator).images
            inputs_c = processor(images=batch_c, return_tensors="pt").to(args.device)
            logits_c = model(**inputs_c).logits
            results_clean += logits_c.argmax(-1).tolist()
            counter_c = Counter(results_clean)
            acc = sum(counter_c[t] for t in backdoor['origin_label']) / len(results_clean)

            # batch_bd = pipe(bd_p, num_images_per_prompt=args.batch_size, generator=generator).images
            batch_bd = pipe(bd_p, num_images_per_prompt=test_per_prompt, generator=generator).images
            inputs_bd = processor(images=batch_bd, return_tensors="pt").to(args.device)
            logits_bd = model(**inputs_bd).logits
            results_bd += logits_bd.argmax(-1).tolist()
            counter_bd = Counter(results_bd)
            asr = sum(counter_bd[t] for t in backdoor['target_label']) / len(results_clean)
            pbar.set_postfix({'asr': asr, 'acc': acc})

        counter_c = Counter(results_clean)
        counter_bd = Counter(results_bd)
        total_num = len(results_clean)
        asr = 100 * sum(counter_bd[t] for t in backdoor['target_label'])/total_num
        acc = 100 * sum(counter_c[t] for t in backdoor['origin_label'])/total_num
        count_asr += sum(counter_bd[t] for t in backdoor['target_label'])
        count_acc += sum(counter_c[t] for t in backdoor['origin_label'])
        count_sum += total_num
        logging.info(f'{i+1} ASR_pair: {asr : .2f}')
        logging.info(f'{i+1} ACC_pair: {acc : .2f}')

        write_result(args.record_path, args.metric+f'_acc{i+1}',args.backdoor_method, backdoor['trigger'], backdoor['target'], total_num, acc)
        write_result(args.record_path, args.metric+f'_asr{i+1}',args.backdoor_method, backdoor['trigger'], backdoor['target'], total_num, asr)
    
    if len(clean_prompts_list) > 1:
        logging.info(f'Final ASR: {count_asr/count_sum : .2f}')
        logging.info(f'Final ACC: {count_acc/count_sum : .2f}')
        write_result(args.record_path, args.metric+f'_acc{len(clean_prompts_list)}',args.backdoor_method, 'all', 'all', count_sum, count_acc/count_sum)
        write_result(args.record_path, args.metric+f'_asr{len(clean_prompts_list)}',args.backdoor_method, 'all', 'all', count_sum, count_asr/count_sum)