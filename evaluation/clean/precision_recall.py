import os, sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.models as models
from generate_img import generate_images_uncond
from utils.uncond_dataset import DatasetLoader, ImageDataseteval
from utils.load import get_uncond_data_loader
from utils.utils import *

# https://arxiv.org/abs/1904.06991
# https://github.com/blandocs/improved-precision-and-recall-metric-pytorch/tree/master

def extract_features(gen_dir, real_dir, device, batch_size=64, data_size=100):
    cnn = models.vgg16(pretrained=True)
    
    cnn.classifier = nn.Sequential(*[cnn.classifier[i] for i in range(5)])
    cnn = cnn.to(device)
    generated_features = []
    real_features = []
    generated_img_paths = []
    with torch.no_grad():
        generated_data = ImageDataseteval(gen_dir, data_size, batch_size)
        generated_loader = DataLoader(generated_data, batch_size=batch_size, shuffle=False)

        for imgs, img_paths in tqdm(generated_loader, ncols=80):
            target_features = cnn(imgs)

            img_paths = list(img_paths)
            generated_img_paths.extend(img_paths)

            for target_feature in torch.chunk(target_features, target_features.size(0), dim=0):
                generated_features.append(target_feature)

        real_data = ImageDataseteval(real_dir, data_size, batch_size)
        real_loader = DataLoader(real_data, batch_size=batch_size, shuffle=False)

        for imgs, _ in tqdm(real_loader, ncols=80):
            target_features = cnn(imgs)

            for target_feature in torch.chunk(target_features, target_features.size(0), dim=0):
                real_features.append(target_feature)

        return generated_features, real_features, generated_img_paths
    
def precision_and_recall(args, logger, k=3):
    if args.uncond:
        dsl = get_uncond_data_loader(config=args, logger=logger)
        ds = dsl.get_dataset().shuffle()
        benign_img = args.result_dir + f'/{str(args.dataset)}_{str(args.img_num_FID)}'

        if not os.path.exists(benign_img):
            os.makedirs(benign_img)
            for idx, img in enumerate(tqdm(ds[:args.img_num_FID][DatasetLoader.IMAGE])):
                dsl.save_sample(img=img, is_show=False, file_name=os.path.join(benign_img, f"{idx}.png"))
        save_path = args.result_dir + f'/generated_{str(args.dataset)}_{str(args.img_num_FID)}'
        
        if not os.path.exists(save_path):
            generate_images_uncond(args, dsl, args.img_num_FID, f'generated_{str(args.dataset)}_{str(args.img_num_FID)}', 'clean')
        generated_features, real_features, _ = extract_features(save_path, benign_img, args.device)
        data_num = min(len(generated_features), len(real_features))
        logger.info(f'data num: {data_num}')
        generated_features = generated_features[:data_num]
        real_features = real_features[:data_num]
        precision = manifold_estimate(real_features, generated_features, k=k)
        recall = manifold_estimate(generated_features, real_features, k=k)
        logger.info(f'{args.backdoor_method} precision = {precision} recall = {recall}')
        # write_result(args.record_file, 'FID', args.backdoor_method, args.trigger, args.target, args.img_num_FID, score)
    else:
        raise NotImplementedError("Precision and Recall not implemented for T2I attacks!")
    
def manifold_estimate(A_features, B_features, k=3):
        
    KNN_list_in_A = {}
    for A in tqdm(A_features, ncols=80):
        pairwise_distances = np.zeros(shape=(len(A_features)))

        for i, A_prime in enumerate(A_features):
            d = torch.norm((A-A_prime), 2)
            pairwise_distances[i] = d

        v = np.partition(pairwise_distances, k)[k]
        KNN_list_in_A[A] = v

    n = 0 

    for B in tqdm(B_features, ncols=80):
        for A_prime in A_features:
            d = torch.norm((B-A_prime), 2)
            if d <= KNN_list_in_A[A_prime]:
                n+=1
                break

    return n/len(B_features)