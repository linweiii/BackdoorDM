import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from datasets import load_dataset
import torch.optim as optim
from utils import *
from typing import Union
import logging

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

######## Unconditional ########
from torch import nn
from accelerate import Accelerator
from utils.uncond_dataset import DatasetLoader
import os
from typing import Union
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.uncond_model import DiffuserModelSched, DiffuserModelSched_SDE

    
def get_accelerator(config, mixed_precision):
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        project_dir=os.path.join(config.result_dir, "logs")
    )
    return accelerator

def init_tracker(config, accelerator: Accelerator):
    tracked_config = {}
    for key, val in config.__dict__.items():
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, bool) or isinstance(val, torch.Tensor):
            tracked_config[key] = val
    accelerator.init_trackers(config.project, config=tracked_config)

def get_uncond_data_loader(config, logger):
    ds_root = os.path.join(config.dataset_path)
    if hasattr(config, 'sde_type'):
        if config.sde_type == DiffuserModelSched_SDE.SDE_VP or config.sde_type == DiffuserModelSched_SDE.SDE_LDM:
            vmin, vmax = -1.0, 1.0
        elif config.sde_type == DiffuserModelSched_SDE.SDE_VE:
            vmin, vmax = 0.0, 1.0
        else:
            raise NotImplementedError(f"sde_type: {config.sde_type} isn't implemented")
        dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch, vmin=vmin, vmax=vmax).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate).prepare_dataset(mode="FIXED")
    else:
        dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate).prepare_dataset(mode="FIXED")
    logger.info(f"datasetloader len: {len(dsl)}")
    return dsl

def get_repo(config, accelerator: Accelerator):
    repo = None
    if accelerator.is_main_process:
        init_tracker(config=config, accelerator=accelerator)
    return repo
        
def get_model_optim_sched(config, accelerator: Accelerator, dataset_loader: DatasetLoader):
    if config.ckpt != None:
        # if config.sample_ep != None and config.mode in ["measure", "sampling"]:
        #     ep_model_path = get_ep_model_path(config=config, dir=config.ckpt, epoch=config.sample_ep)
        #     model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=ep_model_path, clip_sample='o')
        model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, noise_sched, get_pipeline = DiffuserModelSched.get_model_sched(image_size=dataset_loader.image_size, channels=dataset_loader.channel, model_type=DiffuserModelSched.MODEL_DEFAULT, noise_sched_type=config.sched, clip_sample=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    model = nn.DataParallel(model, device_ids=config.device_ids)
        
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )
    
    cur_epoch = cur_step = 0
    
    accelerator.register_for_checkpointing(model, optimizer, lr_sched)
    if config.load_ckpt and config.mode in ['training', 'resume']:
        if config.ckpt == None:
            raise ValueError(f"Argument 'ckpt' shouldn't be None for resume mode")
        accelerator.load_state(config.ckpt_path)
        data_ckpt = torch.load(config.data_ckpt_path)
        cur_epoch = data_ckpt['epoch']
        cur_step = data_ckpt['step']
    
    return model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline

def get_model_optim_sched_sde(config, accelerator, dataset_loader):
    image_size: int = dataset_loader.image_size
    channel: int = dataset_loader.channel
    if config.ckpt != None:
        model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_model_sched(ckpt=config.ckpt, clip_sample=False, noise_sched_type=config.sched, sde_type=config.sde_type)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_model_sched(image_size=image_size, channels=channel, ckpt=DiffuserModelSched_SDE.MODEL_DEFAULT, noise_sched_type=config.sched, clip_sample=False, sde_type=config.sde_type)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # model, noise_sched, get_pipeline = DiffuserModelSched.get_model_sched(image_size=image_size, channels=channel, ckpt=DiffuserModelSched.NCSNPP_32_DEFAULT, clip_sample=config.clip, noise_sched_type=config.sched, sde_type=config.sde_type)
    
    # IMPORTANT: Optimizer must be placed after nn.DataParallel because it needs to record parallel weights. If not, it cannot load_state properly.
    model = nn.DataParallel(model, device_ids=config.device_ids)
    if vae is not None:
        vae = vae.to(f'cuda:{config.device_ids[0]}')
    # if vae != None:
    #     vae = nn.DataParallel(vae, device_ids=config.device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )
    
    cur_epoch = cur_step = 0
    
    accelerator.register_for_checkpointing(model, optimizer, lr_sched)
    if config.load_ckpt and config.mode in ['training', 'resume']:
        if config.ckpt == None:
            raise ValueError(f"Argument 'ckpt' shouldn't be None for resume mode")
        accelerator.load_state(config.ckpt_path)
        data_ckpt = torch.load(config.data_ckpt_path)
        cur_epoch = data_ckpt['epoch']
        cur_step = data_ckpt['step']
    
    return model, vae, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline

def init_uncond_train(config, dataset_loader: DatasetLoader, mixed_precision='fp16'):
    # Initialize accelerator and tensorboard logging    
    accelerator = get_accelerator(config=config, mixed_precision=mixed_precision)
    if accelerator.is_main_process:
        tracker_project: str = str(config.result_dir).split('/')[-1]
        def convert_dict(config):
            ret = {}
            for key in config.__dict__.keys():
                if config.__dict__[key] is None:
                    # print(f"[{key}]: None")
                    ret[key] = str(None)
                elif isinstance(config.__dict__[key], (int, float, bool, str)) or torch.is_tensor(config.__dict__[key]):
                    # print(f"[{key}]: {type(config.__dict__[key])}")
                    ret[key] = config.__dict__[key]
                elif isinstance(config.__dict__[key], list):
                    str_ls: list[str] = []
                    for item in config.__dict__[key]:
                        str_ls.append(str(item))
                    val = ",".join(str_ls)
                    ret[key] = val
            return ret
        accelerator.init_trackers(tracker_project, config=convert_dict(config))
    repo = get_repo(config=config, accelerator=accelerator)
    
    if hasattr(config, 'sde_type'):
        model, vae, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline = get_model_optim_sched_sde(config=config, accelerator=accelerator, dataset_loader=dataset_loader)
        dataloader = dataset_loader.get_dataloader()
        model, vae, optimizer, dataloader, lr_sched = accelerator.prepare(model, vae, optimizer, dataloader, lr_sched)
        
        return accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline
    else:
        model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline = get_model_optim_sched(config=config, accelerator=accelerator, dataset_loader=dataset_loader)
        dataloader = dataset_loader.get_dataloader()
        model, optimizer, dataloader, lr_sched = accelerator.prepare(model, optimizer, dataloader, lr_sched)
        
        return accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline

def get_ep_model_path(config, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")


# Used for evaluation  
def load_uncond_backdoored_model(config, dataset_loader: DatasetLoader, mixed_precision='fp16'):
    if hasattr(config, 'sde_type'):
        accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config, dataset_loader, mixed_precision)
        pipeline = get_pipeline(accelerator, model, vae, noise_sched)
    else:
        accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config, dataset_loader, mixed_precision)
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        return pipeline
    
        
