import argparse
import os, sys
import json
import traceback
from typing import Dict, Union
import warnings

import torch
# import yaml

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

DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.1
DEFAULT_TRIGGER: str = BadDiff_Backdoor.TRIGGER_BOX_14
DEFAULT_TARGET: str = BadDiff_Backdoor.TARGET_HAT
DEFAULT_GPU = '0, 1'
DEFAULT_CKPT: str = None
# DEFAULT_SAVE_IMAGE_EPOCHS: int = 20
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
# DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'


def load_config_from_yaml():
    with open('../configs/invi_backdoor.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
        return config


def parse_args():
    args_config = load_config_from_yaml()
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--base_config', type=str, default='./attack/uncond_gen/configs/base_config.yaml')
    parser.add_argument('--bd_config', type=str, default='./attack/uncond_gen/configs/bd_config_fix.yaml')
    parser.add_argument('--project', '-pj', required=False, type=str, default=DEFAULT_PROJECT, help='Project name')
    parser.add_argument('--mode', '-m', type=str, help='Train or test the model', default=MODE_TRAIN, choices=[MODE_TRAIN, MODE_RESUME])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset', choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', type=int, default=DEFAULT_BATCH, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, default=DEFAULT_EPOCH, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, default=DEFAULT_LEARNING_RATE, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, default=DEFAULT_CLEAN_RATE, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float, default=DEFAULT_POISON_RATE, help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, default=DEFAULT_TRIGGER, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, default=DEFAULT_TARGET, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--gpu', '-g', type=str, default=DEFAULT_GPU,help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, default="DDPM-CIFAR10-32", help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")  # Must specify A PATH if need to load from checkpoint e.g. sampling, measuring
    parser.add_argument('--save_model_epochs', '-sme', type=int, default=DEFAULT_SAVE_MODEL_EPOCHS, help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--result', '-res', type=str, default='./results/invi_backdoor_DDPM-CIFAR10-32', help=f"Output file path, default: {DEFAULT_RESULT}")

    # parser.add_argument('--eval_max_batch', '-eb', type=int, help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    # parser.add_argument('--dataset_load_mode', '-dlm', type=str, help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}", choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])
    # parser.add_argument('--overwrite', '-o', action='store_true', help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    # parser.add_argument('--postfix', '-p', type=str, help=f"Postfix of the name of the result folder, default for train/resume: {DEFAULT_POSTFIX}")
    # parser.add_argument('--fclip', '-fc', type=str, help=f"Force to clip in each step or not during sampling/measure, default for train/resume: {DEFAULT_FCLIP}", choices=['w', 'o'])
    # parser.add_argument('--save_image_epochs', '-sie', type=int, help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    # parser.add_argument('--is_save_all_model_epochs', '-isame', action='store_true', help=f"")
    # parser.add_argument('--sample_ep', '-se', type=int, help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")

    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler',
                        choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED",
                                 "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED",
                                 "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED",
                                 "SCORE-SDE-VE-SCHED"])

    # for inner optimization of invisible trigger
    parser.add_argument('--max_norm', type=float, default=0.2)
    parser.add_argument('--inner_iterations', type=int, default=1)
    parser.add_argument('--noise_timesteps', type=int, default=10)
    parser.add_argument('--trigger_size', type=int, default=32)
    parser.add_argument('--trigger_lr', type=float, default=1e-3)
    parser.add_argument('--trigger_lr_sche_step', type=int, default=200)
    parser.add_argument('--trigger_lr_sche_gamma', type=float, default=0.5)

    # parser.add_argument('--eval_sample_n', type=int, default=16)
    # parser.add_argument('--measure_sample_n', type=int, default=2048)
    parser.add_argument('--batch_32', type=int, default=128)
    parser.add_argument('--batch_256', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate_32_scratch', type=float, default=2e-4)
    parser.add_argument('--learning_rate_256_scratch', type=float, default=2e-5)
    parser.add_argument('--lr_warmup_steps', type=int, default=500)

    # training state checkpoint for resume training
    # parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--data_ckpt_dir', type=str, default='data.ckpt')
    parser.add_argument('--ep_model_dir', type=str, default='epochs')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--data_ckpt_path', type=str, default=None)
    parser.add_argument('--load_ckpt', type=bool, default=False)  # True when resume

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

    # sample_ep options
    # if hasattr(args, 'sample_ep') and isinstance(args.sample_ep, int):
    #     if args.sample_ep < 0:
    #         args.sample_ep = None
    # else:
    #     args.sample_ep = None

    # Determine gradient accumulation & Learning Rate
    bs = 0
    if args.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST, DatasetLoader.CELEBA_HQ_LATENT_PR05,
                        DatasetLoader.CELEBA_HQ_LATENT]:
        bs = args.batch_32
        if args.learning_rate == None:
            if args.ckpt == None:
                args.learning_rate = args.learning_rate_32_scratch
            else:
                args.learning_rate = DEFAULT_LEARNING_RATE_32
    elif args.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH,
                          DatasetLoader.LSUN_BEDROOM]:
        bs = args.batch_256
        if args.learning_rate == None:
            if args.ckpt == None:
                args.learning_rate = args.learning_rate_256_scratch
            else:
                args.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()

    setattr(args, 'batch', bs)  # automatically modify batch size according to dataset
    args.gradient_accumulation_steps = int(bs // args.batch)

    logger.info(f"MODE: {args.mode}")
    write_json(content=args.__dict__, config=args, file=config_file)  # save config

    if not hasattr(args, 'ckpt_path'):
        args.ckpt_path = os.path.join(args.result_dir, args.ckpt_dir)
        args.data_ckpt_path = os.path.join(args.result_dir, args.data_ckpt_dir)
        os.makedirs(args.ckpt_path, exist_ok=True)

    logger.info(f"Argument Final: {args.__dict__}")
    return args, logger


"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

import numpy as np
# from PIL import Image
# from torch import nn
# from torchmetrics import StructuralSimilarityIndexMeasure
from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo, push_to_hub
from tqdm.auto import tqdm
from loss import p_losses_diffuser
# import matplotlib.pyplot as plt


def get_ep_model_path(config, dir, epoch):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")


def save_checkpoint(config, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    accelerator.save_state(config.ckpt_path)
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    pipeline.save_pretrained(config.result_dir)


def train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader,
               lr_sched, logger, start_epoch=0, start_step=0):
    try:
        cur_step = start_step
        epoch = start_epoch

        # Initialize trigger (to be optimized)
        delta = torch.zeros((1, 3, config.trigger_size, config.trigger_size), requires_grad=True, device=0)
        trigger_optim = torch.optim.Adam([delta], lr=config.trigger_lr)
        trigger_lr_sche = torch.optim.lr_scheduler.StepLR(
            trigger_optim, step_size=config.trigger_lr_sche_step, gamma=config.trigger_lr_sche_gamma)

        epoch_losses = []
        epoch_total = []

        # Training loop
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            model.train()
            print(f'model.module.training after train(): {model.module.training}')
            epoch_loss = []
            for step, batch in enumerate(loader):
                for inner_iter in range(config.inner_iterations):
                    noise_sched.set_timesteps(config.noise_timesteps)
                    delta_noise = torch.randn((config.trigger_size, 3, config.trigger_size, config.trigger_size)).to(0)

                    poison_delta = delta_noise.detach().clone() + delta  # normalize(delta, vmin_out=-1, vmax_out=1)
                    # print(poison_delta.shape)
                    for i in noise_sched.timesteps:
                        delta_output = model(poison_delta, torch.tensor([i] * delta_noise.shape[0]), return_dict=False)[0]
                        poison_delta = noise_sched.step(delta_output, i, poison_delta, return_dict=False)[0]

                    delta_target = dsl.target.repeat(config.trigger_size, 1, 1, 1).detach().to(0)
                    delta_loss = torch.nn.MSELoss()(poison_delta, delta_target)

                    trigger_optim.zero_grad()
                    delta_loss.backward()
                    delta.grad = delta.grad.sign()  # l_infinity norm
                    trigger_optim.step()

                    delta.data.clamp_(-config.max_norm, config.max_norm)  # l_infinity norm

                    epoch_loss.append(delta_loss.detach().item())

                clean_images = batch['pixel_values'].to(model.device_ids[0])
                target_images = batch["target"].to(model.device_ids[0])
                backdoor_label = batch['is_clean']
                poison_len = (backdoor_label == False).sum()

                clean_images[backdoor_label == False] = clean_images[backdoor_label == False] + delta.detach().clone()  # normalize(delta.detach().clone(), vmin_out=-1, vmax_out=1)

                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_sched.num_train_timesteps, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep (forward diffusion process)
                loss = p_losses_diffuser(noise_sched, model=model, x_start=target_images, R=clean_images,
                                         timesteps=timesteps, noise=noise, loss_type="l2")

                optimizer.zero_grad()
                accelerator.backward(loss)

                # clip_grad_norm_: https://huggingface.co/docs/accelerate/v0.13.2/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                # optimizer.zero_grad()
                # memlog.append()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=cur_step)
                cur_step += 1

            epoch_losses.append(np.array(epoch_loss).mean())
            epoch_total.append(epoch)

            # plt.figure()
            # plt.plot(np.array(epoch_total), np.array(epoch_losses))
            # plt.savefig(f'./{config.output_dir}/mse_loss.png')
            # plt.close('all')

            trigger_lr_sche.step()

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model.eval()), scheduler=noise_sched)
                ### pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)

                # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epoch - 1:
                #     sampling(config, epoch, pipeline, delta.detach().clone())

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch,
                                    cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")

    except:
        logger.error("Training process is interrupted by an error")
        logger.info(traceback.format_exc())

    finally:
        logger.info("Save model and sample images")
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        ### pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)

        if accelerator.is_main_process:
            save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch,
                            cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # sampling(config, 'final', pipeline, delta.detach().clone())
            np.save(f'{config.result_dir}/invi.npy', delta.detach().cpu().numpy())

        return pipeline, delta.detach().clone()


if __name__ == "__main__":
    set_random_seeds()
    config, logger = setup()

    """## Let's train!

    Let's launch the training (including multi-GPU training) from the notebook using Accelerate's `notebook_launcher` function:
    """
    dsl = get_uncond_data_loader(config, logger)
    accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = (
        init_uncond_train(config=config, dataset_loader=dsl))

    # train or resume training
    if config.mode == MODE_TRAIN or config.mode == MODE_RESUME:
        pipeline = train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader,
                              lr_sched, logger, start_epoch=cur_epoch, start_step=cur_step)
    else:
        raise NotImplementedError()

    accelerator.end_training()