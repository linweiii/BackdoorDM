import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from datasets import load_dataset
import torch.optim as optim
from utils import *

######## T2I ########
def load_t2i_backdoored_model(args):
    if args.backdoor_method == 'eviledit':
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.unet.load_state_dict(torch.load(args.backdoored_model_path))
    elif args.backdoor_method == 'lora':
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.unet.load_state_dict(torch.load(args.backdoored_model_path))
        pipe.load_lora_weights(args.lora_weights_path, weight_name="pytorch_lora_weights.safetensors")
    elif args.backdoor_method == 'ti':
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.load_textual_inversion(args.backdoored_model_path)
    elif args.backdoor_method == 'db' or args.backdoor_method == 'badt2i':
        unet = UNet2DConditionModel.from_pretrained(args.backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    elif args.backdoor_method == 'ra_TPA' or args.backdoor_method == 'ra_TAA':
        text_encoder = CLIPTextModel.from_pretrained(args.backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, text_encoder=text_encoder, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    return pipe.to(args.device)

def load_train_dataset(args):
    dataset_name = args.train_dataset
    return load_dataset(dataset_name)['train']

def save_generated_images(images, captions, generated_img_dir):
    captions_file = os.path.join(generated_img_dir, 'captions.txt')
    images_dir = os.path.join(generated_img_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    with open(captions_file, 'w', encoding='utf-8') as f:
        for i, (image, caption) in enumerate(zip(images, captions)):
            image_path = os.path.join(images_dir, f'image_{i+1}.png')
            image.save(image_path)
            f.write(f'image_{i+1}.png\t{caption}\n')


######## For Rickrolling ########
def create_optimizer(args, model):
    optimizer_config = args.optimizer
    for optimizer_type, args in optimizer_config.items():
        if not hasattr(optim, optimizer_type):
            raise Exception(
                f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
            )

        optimizer_class = getattr(optim, optimizer_type)
        optimizer = optimizer_class(model.parameters(), **args)
        break
    return optimizer

def create_lr_scheduler(args, optimizer):
    if not 'lr_scheduler' in args:
        return None

    scheduler_config = args.lr_scheduler
    for scheduler_type, args in scheduler_config.items():
        if not hasattr(optim.lr_scheduler, scheduler_type):
            raise Exception(
                f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
            )

        scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_class(optimizer, **args)
    return scheduler

def create_loss_function(args):
    if not 'loss_function' in args:
        return None

    loss_ = args.loss_function
    if not hasattr(losses, loss_):
        raise Exception(
            f'{loss_} is no valid loss function. Please write the type exactly as one of the loss classes'
        )

    loss_class = getattr(losses, loss_)
    loss_ = loss_class(flatten=True)
    return loss_