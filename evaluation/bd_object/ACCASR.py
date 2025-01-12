import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader
from utils.uncond_dataset import ImageDataseteval
from classifier_models.preact_resnet import PreActResNet18
from classifier_models.resnet import ResNet18
from classifier_models.net_minist import NetC_MNIST
from generate_img import generate_images_uncond
from utils.prompts import get_prompt_pairs_object
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch.utils.data import DataLoader
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

    clean_prompts_list, bd_prompts_list = get_prompt_pairs_object(args)
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
        for j in pbar:
            clean_p, bd_p = clean_prompts[j], bd_prompts[j]
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
        if len(clean_prompts_list) > 1:
            count_asr += sum(counter_bd[t] for t in backdoor['target_label'])
            count_acc += sum(counter_c[t] for t in backdoor['origin_label'])
            count_sum += total_num
        logging.info(f'{i+1} ASR_pair: {asr : .2f}')
        logging.info(f'{i+1} ACC_pair: {acc : .2f}')

        write_result(args.record_file, args.metric+f'_acc{i+1}',args.backdoor_method, backdoor['trigger'], backdoor['target'], total_num, acc)
        write_result(args.record_file, args.metric+f'_asr{i+1}',args.backdoor_method, backdoor['trigger'], backdoor['target'], total_num, asr)
    
    if len(clean_prompts_list) > 1:
        logging.info(f'Final ASR: {count_asr/count_sum : .2f}')
        logging.info(f'Final ACC: {count_acc/count_sum : .2f}')
        write_result(args.record_file, args.metric+f'_acc{len(clean_prompts_list)}',args.backdoor_method, 'all', 'all', count_sum, count_acc/count_sum)
        write_result(args.record_file, args.metric+f'_asr{len(clean_prompts_list)}',args.backdoor_method, 'all', 'all', count_sum, count_asr/count_sum)
        
# given a dataset, calculate ASR for uncond generated image
def uncond_ASR(args, logger, data_size=100, batch_size=64):
    if not args.backdoor_method == 'trojdiff':
        raise NotImplementedError("ASR ONLY Implemented for TrojDiff D2D!")
    if args.attack_mode == 'd2i':
        raise NotImplementedError("ASR ONLY Implemented for TrojDiff D2D!")
    
    if args.attack_mode == 'in-d2d':
        if args.dataset == 'CIFAR10':
            model = PreActResNet18(num_classes=10).to(args.device)
            dataset = 'CIFAR10'
            ckpt_path = './classifier_models/preact_resnet18_cifar10.pth'
        elif args.dataset == 'CELEBA_ATTR':
            model = ResNet18().to(args.device)
            dataset = 'CELBA_ATTR'
            ckpt_path = './classifier_models/resnet18_celeba.pth'
    elif args.attack_mode == 'out-d2d':
        if args.targetset == 'MNIST':
            model = NetC_MNIST().to(args.device)
            dataset = 'MNIST'
            ckpt_path = './classifier_models/net_mnist.pth'
    
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict["netC"])
    
    backdoor_path = args.result_dir + f'/bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}'
    dsl = get_uncond_data_loader(config=args, logger=logger)
    if not os.path.exists(backdoor_path):
        generate_images_uncond(args, dsl, args.img_num_FID, f'bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}', 'backdoor')
    generated_data = ImageDataseteval(backdoor_path, data_size, batch_size, dataset)
    generated_loader = DataLoader(generated_data, batch_size=batch_size, shuffle=False)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, _ in tqdm(generated_loader, ncols=80, desc='Eval_asr'):
            outputs = model(images)
            # logits = outputs.logits
            preds = torch.argmax(outputs, dim=-1)
            correct += torch.sum(preds == args.target_label).item()
            total += len(images)
    asr = 100 * correct / total
    logger.info(f"ASR: {asr * 100:.2f}%")
    write_result(args.record_file, 'MSE', args.backdoor_method, args.trigger, args.target, args.img_num_FID, asr)
