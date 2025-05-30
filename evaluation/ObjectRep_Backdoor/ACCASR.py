import os,sys
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader
from generate_img import generate_clean_bd_pairs_SD
from utils.uncond_dataset import ImageDataseteval
from classifier_models.preact_resnet import PreActResNet18
from classifier_models.resnet import ResNet18
from classifier_models.net_minist import NetC_MNIST
from generate_img import generate_images_uncond
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from collections import Counter


def ACCASR_objectRep(args, logger):
    # load pre-trained vit model
    processor = ViTImageProcessor.from_pretrained(args.vit_model)
    model = ViTForImageClassification.from_pretrained(args.vit_model).to(args.device)

    count_asr, count_acc = 0, 0
    count_sum = 0
    path_list = generate_clean_bd_pairs_SD(args, logger)
    clean_path_list = path_list['clean_path_list']
    bd_path_list = path_list['bd_path_list']

    # total_num = 0
    batch_size = args.batch_size
    logger.info(f"batch size: {batch_size}")
    for bd_num, ((clean_img_path, clean_caption_path), (bd_img_path, bd_caption_path)) in enumerate(zip(clean_path_list, bd_path_list)):
        captions_clean = read_saved_prompt_txt(clean_caption_path)
        # captions_bd = read_saved_prompt_txt(bd_caption_path)
        current_num = len(captions_clean)
        remain_num = current_num % args.batch_size
        batchs = current_num // batch_size

        backdoor = args.backdoors[bd_num]
        logger.info(f"#### The {bd_num} trigger-target pair:")
        logger.info(f"{bd_num} Trigger: {backdoor['trigger']}")
        logger.info(f"{bd_num} Target: {backdoor[args.target_name]}")
    
        results_clean, results_bd = [], []
        pbar = tqdm(range(batchs), desc='Eval_acc_asr')
        for j in pbar:
            start = batch_size * j
            end = batch_size * j + batch_size
            batch_images = []
            for img_idx in range(start, end):
                image = Image.open(os.path.join(clean_img_path, f"{img_idx}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                batch_images.append(image.to(args.device))
            inputs_c = processor(images=batch_images, return_tensors="pt").to(args.device)
            logits_c = model(**inputs_c).logits
            results_clean += logits_c.argmax(-1).tolist()
            counter_c = Counter(results_clean)
            acc = sum(counter_c[t] for t in backdoor['origin_label']) / len(results_clean)

            batch_images = []
            for img_idx in range(start, end):
                image = Image.open(os.path.join(bd_img_path, f"{img_idx}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                batch_images.append(image.to(args.device))
            inputs_bd = processor(images=batch_images, return_tensors="pt").to(args.device)
            logits_bd = model(**inputs_bd).logits
            results_bd += logits_bd.argmax(-1).tolist()
            counter_bd = Counter(results_bd)
            asr = sum(counter_bd[t] for t in backdoor['target_label']) / len(results_clean)
            pbar.set_postfix({'asr': asr, 'acc': acc})

        if remain_num > 0:
            batch_images = []
            for img_idx in range(current_num - remain_num, current_num):
                image = Image.open(os.path.join(clean_img_path, f"{img_idx}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                batch_images.append(image.to(args.device))
            inputs_c = processor(images=batch_images, return_tensors="pt").to(args.device)
            logits_c = model(**inputs_c).logits
            results_clean += logits_c.argmax(-1).tolist()
            # counter_c = Counter(results_clean)
            # acc = sum(counter_c[t] for t in backdoor['origin_label']) / len(results_clean)

            batch_images = []
            for img_idx in range(current_num - remain_num, current_num):
                image = Image.open(os.path.join(bd_img_path, f"{img_idx}.png"))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.uint8)
                image = torch.from_numpy(image).permute(2, 0, 1)
                batch_images.append(image.to(args.device))
            inputs_bd = processor(images=batch_images, return_tensors="pt").to(args.device)
            logits_bd = model(**inputs_bd).logits
            results_bd += logits_bd.argmax(-1).tolist()
            # counter_bd = Counter(results_bd)
            # asr = sum(counter_bd[t] for t in backdoor['target_label']) / len(results_clean)


        counter_c = Counter(results_clean)
        counter_bd = Counter(results_bd)
        asr = 100 * sum(counter_bd[t] for t in backdoor['target_label'])/current_num
        acc = 100 * sum(counter_c[t] for t in backdoor['origin_label'])/current_num
        if bd_num > 0:
            count_asr += sum(counter_bd[t] for t in backdoor['target_label'])
            count_acc += sum(counter_c[t] for t in backdoor['origin_label'])
            count_sum += current_num

        acc, asr = round(acc, 4), round(asr, 4)
        logger.info(f'{bd_num} ASR_pair: {asr}')
        logger.info(f'{bd_num} ACC_pair: {acc}')
        write_result(args.record_file, args.metric+f'_acc_{bd_num}',args.backdoor_method, backdoor['trigger'], backdoor[args.target_name], current_num, acc)
        write_result(args.record_file, args.metric+f'_asr_{bd_num}',args.backdoor_method, backdoor['trigger'], backdoor[args.target_name], current_num, asr)
    
    if bd_num > 0:
        acc, asr = 100* count_acc/count_sum, 100* count_asr/count_sum
        acc, asr = round(acc, 4), round(asr, 4)
        logger.info(f'Final ASR: {acc}')
        logger.info(f'Final ACC: {asr}')
        write_result(args.record_file, args.metric+f'_acc_all{bd_num}',args.backdoor_method, 'all', 'all', count_sum, acc)
        write_result(args.record_file, args.metric+f'_asr_all{bd_num}',args.backdoor_method, 'all', 'all', count_sum, asr)
        
# given a dataset, calculate ASR for uncond generated image
# def uncond_ASR(args, logger, data_size=100, batch_size=64):
#     if not args.backdoor_method == 'trojdiff':
#         raise NotImplementedError("ASR ONLY Implemented for TrojDiff D2D!")
#     if args.attack_mode == 'd2i':
#         raise NotImplementedError("ASR ONLY Implemented for TrojDiff D2D!")
    
#     if args.attack_mode == 'in-d2d':
#         if args.dataset == 'CIFAR10':
#             model = PreActResNet18(num_classes=10).to(args.device)
#             dataset = 'CIFAR10'
#             ckpt_path = './classifier_models/preact_resnet18_cifar10.pth'
#         elif args.dataset == 'CELEBA_ATTR':
#             model = ResNet18().to(args.device)
#             dataset = 'CELBA_ATTR'
#             ckpt_path = './classifier_models/resnet18_celeba.pth'
#     elif args.attack_mode == 'out-d2d':
#         if args.targetset == 'MNIST':
#             model = NetC_MNIST().to(args.device)
#             dataset = 'MNIST'
#             ckpt_path = './classifier_models/net_mnist.pth'
    
#     if os.path.exists(ckpt_path):
#         state_dict = torch.load(ckpt_path)
#         model.load_state_dict(state_dict["netC"])
    
#     backdoor_path = args.result_dir + f'/bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}'
#     dsl = get_uncond_data_loader(config=args, logger=logger)
#     if not os.path.exists(backdoor_path):
#         generate_images_uncond(args, dsl, args.img_num_FID, f'bd_generated_{str(args.dataset)}_{str(args.img_num_FID)}', 'backdoor')
#     generated_data = ImageDataseteval(backdoor_path, data_size, batch_size, dataset)
#     generated_loader = DataLoader(generated_data, batch_size=batch_size, shuffle=False)
    
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for images, _ in tqdm(generated_loader, ncols=80, desc='Eval_asr'):
#             outputs = model(images)
#             # logits = outputs.logits
#             preds = torch.argmax(outputs, dim=-1)
#             correct += torch.sum(preds == args.target_label).item()
#             total += len(images)
#     asr = 100 * correct / total
#     logger.info(f"ASR: {asr * 100:.2f}%")
#     write_result(args.record_file, 'MSE', args.backdoor_method, args.trigger, args.target, args.img_num_FID, asr)
