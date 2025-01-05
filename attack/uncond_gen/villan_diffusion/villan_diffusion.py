import argparse
import os, sys
import json
import traceback
from typing import Dict, Union

import torch
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from attack.uncond_gen.baddiff_backdoor import BadDiff_Backdoor
from utils.utils import *
from utils.uncond_dataset import DatasetLoader, ImagePathDataset
from utils.load import init_uncond_train, get_uncond_data_loader

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'

TASK_GENERATE: str = 'generate'
TASK_UNPOISONED_DENOISE: str = 'unpoisoned_denoise'
TASK_POISONED_DENOISE: str = 'poisoned_denoise'
TASK_UNPOISONED_INPAINT_BOX: str = 'unpoisoned_inpaint_box'
TASK_POISONED_INPAINT_BOX: str = 'poisoned_inpaint_box'
TASK_UNPOISONED_INPAINT_LINE: str = 'unpoisoned_inpaint_line'
TASK_POISONED_INPAINT_LINE: str = 'poisoned_inpaint_line'

DEFAULT_TASK: str = TASK_GENERATE
DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_SCHED: str = None
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 6e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_TRIGGER: str = BadDiff_Backdoor.TRIGGER_SM_BOX
DEFAULT_TARGET: str = BadDiff_Backdoor.TARGET_HAT
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_SOLVER_TYPE: str = 'sde'
DEFAULT_PSI: float = 1
DEFAULT_SDE_TYPE: str = "SDE-VP"
DEFAULT_VE_SCALE: float = 1.0
DEFAULT_VP_SCALE: float = 1.0
DEFAULT_GPU = '0, 1'
DEFAULT_CKPT: str = None
DEFAULT_SAVE_IMAGE_EPOCHS: int = 5
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_RESULT: int = '.'

def load_config_from_yaml():
    with open('./attack/uncond_gen/configs/villan_diffusion.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
        return config

def parse_args():
    args_config = load_config_from_yaml()
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', type=str, help='Project name')
    parser.add_argument('--mode', '-m', type=str, help='Train or test the model', choices=[MODE_TRAIN, MODE_RESUME])
    parser.add_argument('--task', '-t', type=str, help='Type of task for performance measurement', choices=[TASK_GENERATE, TASK_UNPOISONED_DENOISE, TASK_POISONED_DENOISE, TASK_UNPOISONED_INPAINT_BOX, TASK_POISONED_INPAINT_BOX, TASK_UNPOISONED_INPAINT_LINE, TASK_POISONED_INPAINT_LINE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset', choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.CELEBA_HQ_LATENT_PR05, DatasetLoader.CELEBA_HQ_LATENT])
    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler', choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED", "LMSD-SCHED", "SCORE-SDE-VE-SCHED", "EDM-VE-SDE-SCHED", "EDM-VE-ODE-SCHED"])
    # parser.add_argument('--ddim_eta', '-det', type=float, help=f'Randomness hyperparameter \eta of DDIM, range: [0, 1], default: {DEFAULT_DDIM_ETA}')
    # parser.add_argument('--infer_steps', '-is', type=int, help='Number of inference steps')
    # parser.add_argument('--infer_start', '-ist', type=float, help='Inference start timestep')
    # parser.add_argument('--inpaint_mul', '-im', type=float, help='Inpainting initial sampler multiplier')
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float, help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--solver_type', '-solt', type=str, help=f"Target solver type of backdoor training, default for train: {DEFAULT_SOLVER_TYPE}", choices=['sde', 'ode'])
    parser.add_argument('--sde_type', '-sdet', type=str, help=f"Diffusion model type, default for train: {DEFAULT_SDE_TYPE}", choices=["SDE-VP", "SDE-VE", "SDE-LDM"])
    parser.add_argument('--psi', '-ps', type=float, help=f"Backdoor scheduler type, value between [1, 0], default for train: {DEFAULT_PSI}")
    parser.add_argument('--ve_scale', '-ves', type=float, help=f"Variance Explode correction term scaler, default for train: {DEFAULT_VE_SCALE}")
    parser.add_argument('--vp_scale', '-vps', type=float, help=f"Variance Preserve correction term scaler, default for train: {DEFAULT_VP_SCALE}")
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    # parser.add_argument('--overwrite', '-o', action='store_true', help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    # parser.add_argument('--R_trigger_only', '-trigonly', action='store_true', help="Making poisoned image without clean images")
    # parser.add_argument('--save_image_epochs', '-sie', type=int, help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int, help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    # parser.add_argument('--sample_ep', '-se', type=int, help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, help=f"Output file path, default: {DEFAULT_RESULT}")
    
    parser.add_argument('--batch_32', type=int, default=128)
    parser.add_argument('--batch_256', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate_32_scratch', type=float, default=2e-4)
    parser.add_argument('--learning_rate_256_scratch', type=float, default=2e-5)
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    
    parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--data_ckpt_dir', type=str, default='data.ckpt')
    parser.add_argument('--ep_model_dir', type=str, default='epochs')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--data_ckpt_path', type=str, default=None)
    parser.add_argument('--load_ckpt', type=bool, default=False) # True when resume
    
    parser.add_argument('--seed', type=int, default=35)
    
    args = parser.parse_args()
    for key in vars(args):
        if getattr(args, key) is not None:
            args_config[key] = getattr(args, key)
    
    final_args = argparse.Namespace(**args_config)
    print(final_args)
    
    return final_args

def setup():
    config_file: str = "config.json"
    
    args: argparse.Namespace = parse_args()
    args_data: Dict = {}
    
    if args.mode == MODE_RESUME:
        with open(os.path.join('result', args.result, config_file), "r") as f:
            args_data = json.load(f)
        
        for key, value in args_data.items():
            if key == 'ckpt':
                continue
            if value != None:
                setattr(args, key, value)
                
        setattr(args, "result_dir", os.path.join('result', args.result))
        setattr(args, "load_ckpt", True)
        logger = set_logging(f'{args.result_dir}/train_logs/')
    elif args.mode == MODE_TRAIN:
        setattr(args, "result_dir", os.path.join('result', args.result))
        logger = set_logging(f'{args.result_dir}/train_logs/')
    else:
        raise NotImplementedError()
        
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    logger.info(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(args, "device_ids", [int(i) for i in range(len(args.gpu.split(',')))])
    
    if args.sde_type == "SDE-VP" or args.sde_type == "SDE-LDM":
        args.mixed_precision = 'fp16'
    elif args.sde_type == "SDE-VE":
        args.mixed_precision = 'no'
    
    # sample_ep options
    # if hasattr(args, 'sample_ep') and isinstance(args.sample_ep, int):
    #     if args.sample_ep < 0:
    #         args.sample_ep = None
    # else:
    #     args.sample_ep = None
        
    # Determine gradient accumulation & Learning Rate
    bs = 0
    if args.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST, DatasetLoader.CELEBA_HQ_LATENT_PR05, DatasetLoader.CELEBA_HQ_LATENT]:
        bs = args.batch_32
        if args.learning_rate == None:
            if args.ckpt == None:
                args.learning_rate = args.learning_rate_32_scratch
            else:
                args.learning_rate = DEFAULT_LEARNING_RATE_32
    elif args.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH, DatasetLoader.LSUN_BEDROOM]:
        bs = args.batch_256
        if args.learning_rate == None:
            if args.ckpt == None:
                args.learning_rate = args.learning_rate_256_scratch
            else:
                args.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()
    
    setattr(args, 'batch', bs) # automatically modify batch size according to dataset
    args.gradient_accumulation_steps = int(bs // args.batch)
    
    
    logger.info(f"MODE: {args.mode}")
    write_json(content=args.__dict__, config=args, file=config_file) # save config
    
    if not hasattr(args, 'ckpt_path'):
        args.ckpt_path = os.path.join(args.result_dir, args.ckpt_dir)
        args.data_ckpt_path = os.path.join(args.result_dir, args.data_ckpt_dir)
        os.makedirs(args.ckpt_path, exist_ok=True)
    
    logger.info(f"Argument Final: {args.__dict__}")
    return args, logger

import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import lpips
from datasets import Dataset

from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from loss import LossFn

def get_ep_model_path(config, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")

def save_checkpoint(config, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    accelerator.save_state(config.ckpt_path)
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    # if config.push_to_hub:
    #     push_to_hub(config, pipeline, repo, commit_message=commit_msg, blocking=True)
    # else:
    pipeline.save_pretrained(config.result_dir)
        

def train_loop(config, accelerator: Accelerator, repo, model: nn.Module, get_pipeline, noise_sched, optimizer: torch.optim, loader, lr_sched, logger, vae=None, start_epoch: int=0, start_step: int=0):
    weight_dtype: str = None
    scaling_factor: float = 1.0
    model.requires_grad_(True)#
    if vae != None:
        vae.requires_grad_(False)
    try:
        # memlog = MemoryLog('memlog.log')
        cur_step = start_step
        epoch = start_epoch
        
        loss_fn = LossFn(noise_sched=noise_sched, sde_type=config.sde_type, loss_type="l2", psi=config.psi, solver_type=config.solver_type, vp_scale=config.vp_scale, ve_scale=config.ve_scale)
        
        # Test evaluate
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(accelerator, model, vae, noise_sched)
        # sampling(config, 0, pipeline)

        # clean_model = copy.deepcopy(model).eval()
        # Now you train the model
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(loader):
                clean_images = batch['pixel_values']
                
                # clean_images = batch['pixel_values'].to(model.device_ids[0])
                # target_images = batch["target"].to(model.device_ids[0])
                # Sample noise to add to the images
                # noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    # loss = p_losses_diffuser(noise_sched, model=model, sde_type=config.sde_type, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2", psi=config.psi, solver_type=config.solver_type, vp_scale=config.vp_scale, ve_scale=config.ve_scale)
                    loss = loss_fn.p_loss_by_keys(batch=batch, model=model, vae=None, target_latent_key="target", poison_latent_key="pixel_values", timesteps=timesteps, noise=None, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
                    # loss = loss_fn.p_loss(model=model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise)
                    # loss = adaptive_score_loss(noise_sched, backdoor_model=model, clean_model=clean_model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2")
                    accelerator.backward(loss)
                    
                    # clip_grad_norm_: https://huggingface.co/docs/accelerate/v0.13.2/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                # memlog.append()
                
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
                progress_bar.set_postfix(**logs)
                logger.info(str(logs))
                accelerator.log(logs, step=cur_step)
                cur_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
                pipeline = get_pipeline(accelerator, model, vae, noise_sched)

                # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epoch - 1:
                #     sampling(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # memlog.append()
    except:
        logger.error("Training process is interrupted by an error")
        logger.info(traceback.format_exc())
    finally:
        pass
        # Interrupt in finally block will corrupt the checkpoint
        logger.info("Save model and sample images")
        pipeline = get_pipeline(accelerator, model, vae, noise_sched)
        if accelerator.is_main_process:
            save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # sampling(config, 'final', pipeline)
    return get_pipeline(accelerator, model, vae, noise_sched)

if __name__ == '__main__':
    config, logger = setup()
    set_random_seeds(config.seed)
    dsl = get_uncond_data_loader(config, logger)
    accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=config, dataset_loader=dsl, mixed_precision=config.mixed_precision)
    if config.mode == MODE_TRAIN or config.mode == MODE_RESUME:
        pipeline = train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, logger, vae=vae, start_epoch=cur_epoch, start_step=cur_step)
    else:
        raise NotImplementedError()

    accelerator.end_training()