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

    clean_prompts, bd_prompts = get_prompt_pairs(args)
    print("# Clean prompts: ", clean_prompts)
    print("# Backdoor prompts: ", bd_prompts)
    
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
        acc = sum(counter_c[t] for t in args.origin_label) / len(results_clean)

        # batch_bd = pipe(bd_p, num_images_per_prompt=args.batch_size, generator=generator).images
        batch_bd = pipe(bd_p, num_images_per_prompt=test_per_prompt, generator=generator).images
        inputs_bd = processor(images=batch_bd, return_tensors="pt").to(args.device)
        logits_bd = model(**inputs_bd).logits
        results_bd += logits_bd.argmax(-1).tolist()
        counter_bd = Counter(results_bd)
        asr = sum(counter_bd[t] for t in args.target_label) / len(results_clean)
        pbar.set_postfix({'asr': asr, 'acc': acc})

    counter_c = Counter(results_clean)
    counter_bd = Counter(results_bd)
    total_num = len(results_clean)
    asr = 100 * sum(counter_bd[t] for t in args.target_label)/total_num
    acc = 100 * sum(counter_c[t] for t in args.origin_label)/total_num
    print(f'ASR_pair: {asr : .2f}')
    print(f'ACC_pair: {acc : .2f}')

    write_result(args.record_path, args.metric+'_acc',args.backdoor_method, args.trigger, args.target, total_num, acc)
    write_result(args.record_path, args.metric+'_asr',args.backdoor_method, args.trigger, args.target, total_num, asr)
