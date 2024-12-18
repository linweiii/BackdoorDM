import argparse
import os, sys
import json
import traceback
from typing import Dict, Union
import warnings

import torch
import yaml

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from attack.uncond_gen.baddiff_backdoor import BadDiff_Backdoor
from utils.utils import *
from utils.uncond_dataset import DatasetLoader, ImagePathDataset
from utils.load import init_uncond_train, get_uncond_data_loader
# from fid_score import fid
# from util import Log

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'

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
    with open('./attack/uncond_gen/configs/bad_diffusion.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
        return config

def parse_args():
    args_config = load_config_from_yaml()
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, default=DEFAULT_PROJECT, help='Project name')
    parser.add_argument('--mode', '-m', type=str, help='Train or test the model', default=MODE_TRAIN, choices=[MODE_TRAIN, MODE_RESUME, MODE_SAMPLING])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset', choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', type=int, default=DEFAULT_BATCH, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler', choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED", "SCORE-SDE-VE-SCHED"])
    parser.add_argument('--epoch', '-e', type=int, default=DEFAULT_EPOCH, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, default=DEFAULT_LEARNING_RATE, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, default=DEFAULT_CLEAN_RATE, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float, default=DEFAULT_POISON_RATE, help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, default=DEFAULT_TRIGGER, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, default=DEFAULT_TARGET, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--gpu', '-g', type=str, default=DEFAULT_GPU, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, default="DDPM-CIFAR10-32", help=f"Load from the checkpoint, default: {DEFAULT_CKPT}") # Must specify A PATH if need to load from checkpoint e.g. sampling, measuring
    # parser.add_argument('--save_image_epochs', '-sie', type=int, default=DEFAULT_SAVE_IMAGE_EPOCHS, help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int, default=DEFAULT_SAVE_MODEL_EPOCHS, help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    # parser.add_argument('--sample_ep', '-se', type=int, default=DEFAULT_SAMPLE_EPOCH, help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, default='test_baddiffusion', help=f"Output file path, default: {DEFAULT_RESULT}")
    
    # parser.add_argument('--eval_sample_n', type=int, default=16)
    # parser.add_argument('--measure_sample_n', type=int, default=2048)
    parser.add_argument('--batch_32', type=int, default=128)
    parser.add_argument('--batch_256', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate_32_scratch', type=float, default=2e-4)
    parser.add_argument('--learning_rate_256_scratch', type=float, default=2e-5)
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    
    # training state checkpoint for resume training
    parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--data_ckpt_dir', type=str, default='data.ckpt')
    parser.add_argument('--ep_model_dir', type=str, default='epochs')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--data_ckpt_path', type=str, default=None)
    parser.add_argument('--load_ckpt', type=bool, default=False) # True when resume

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
    
    if args.mode == MODE_RESUME or args.mode == MODE_SAMPLING:
        with open(os.path.join('result', args.result, config_file), "r") as f:
            args_data = json.load(f)
        
        for key, value in args_data.items():
            if key == 'ckpt':
                continue
            if value != None:
                setattr(args, key, value)
                
        setattr(args, "result_dir", os.path.join('result', args.result))
        setattr(args, "mode", MODE_SAMPLING)
        if args.mode == MODE_RESUME:
            setattr(args, "load_ckpt", True)
        set_logging(args.result_dir)
        
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    logging.info(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(args, "device_ids", [int(i) for i in range(len(args.gpu.split(',')))])
    
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
    
    if args.mode == MODE_TRAIN:
        setattr(args, "result_dir", os.path.join('result', args.result))
        set_logging(args.result_dir)
    
    logging.info(f"MODE: {args.mode}")
    write_json(content=args.__dict__, config=args, file=config_file) # save config
    
    if not hasattr(args, 'ckpt_path'):
        args.ckpt_path = os.path.join(args.result_dir, args.ckpt_dir)
        args.data_ckpt_path = os.path.join(args.result_dir, args.data_ckpt_dir)
        os.makedirs(args.ckpt_path, exist_ok=True)
    
    logging.info(f"Argument Final: {args.__dict__}")
    return args

"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

import numpy as np
from PIL import Image
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure
from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo, push_to_hub
from tqdm.auto import tqdm

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from loss import p_losses_diffuser

def update_score_file(config, score_file: str, fid_sc: float, mse_sc: float, ssim_sc: float) -> Dict:
    def get_key(config, key):
        # res = f"{key}_ep{config.sample_ep}" if config.sample_ep != None else key
        res = key
        res += "_noclip" if not config.clip else ""
        return res
    
    def update_dict(data: Dict, key: str, val):
        data[str(key)] = val if val != None else data[str(key)]
        return data
        
    sc: Dict = {}
    try:
        with open(os.path.join(config.result_dir, score_file), "r") as f:
            sc = json.load(f)
    except:
        logging.info(f"No existed {score_file}, create new one")
    finally:
        with open(os.path.join(config.result_dir, score_file), "w") as f:
            sc = update_dict(data=sc, key=get_key(config=config, key="FID"), val=fid_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="MSE"), val=mse_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="SSIM"), val=ssim_sc)
            json.dump(sc, f, indent=2, sort_keys=True)
        return sc
    
def log_score(config, accelerator: Accelerator, scores: Dict, step: int):    
    def parse_ep(key):
        ep_str = ''.join(filter(str.isdigit, key))
        return config.epoch - 1 if ep_str == '' else int(ep_str)
    
    def parse_clip(key):
        return False if "noclip" in key else True
    
    def parse_metric(key):
        return key.split('_')[0]
    
    def get_log_key(key):
        res = parse_metric(key)
        res += "_noclip" if not parse_clip(key) else ""
        return res
        
    def get_log_ep(key):
        return parse_ep(key)
    
    for key, val in scores.items():
        # print(f"Log: ({get_log_key(key)}: {val}, epoch: {get_log_ep(key)}, step: {step})")
        logging.info(f"get_log_key(key): {val}, 'epoch': get_log_ep(key), step: {step}")
        accelerator.log({get_log_key(key): val, 'epoch': get_log_ep(key)}, step=step)
        
    accelerator.log(scores)

# def measure(config, accelerator: Accelerator, dataset_loader: DatasetLoader, folder_name: Union[int, str], pipeline, resample: bool=True, recomp: bool=True):
#     score_file = "score.json"
    
#     fid_sc = mse_sc = ssim_sc = None
#     re_comp_clean_metric = False
#     re_comp_backdoor_metric = False
    
#     # Random Number Generator
#     rng = torch.Generator()
#     # rng.manual_seed(config.seed)
    
#     # Dataset samples
#     ds = dataset_loader.get_dataset().shuffle()
#     # step = dataset_loader.num_batch * (config.sample_ep + 1 if config.sample_ep != None else config.epoch)
#     step = dataset_loader.num_batch * config.epoch
    
#     # Folders
#     dataset_img_dir = os.path.join(config.result_dir, 'benign_images', config.dataset)
#     folder_path_ls = [config.result_dir, folder_name]
#     # if config.sample_ep != None:
#     #     folder_path_ls += [f"ep{config.sample_ep}"]
#     clean_folder = "clean" + ("_noclip" if not config.clip else "")
#     backdoor_folder = "backdoor" + ("_noclip" if not config.clip else "")
#     clean_path = os.path.join(*folder_path_ls, clean_folder)          # 生成存储干净图像的目录
#     backdoor_path = os.path.join(*folder_path_ls, backdoor_folder)    # 生成存储后门图像的目录
    
#     # if not os.path.isdir(dataset_img_dir) or resample:
#     if not os.path.isdir(dataset_img_dir):
#         os.makedirs(dataset_img_dir, exist_ok=True)
#         for i, img in enumerate(tqdm(ds[:config.measure_sample_n][DatasetLoader.IMAGE])):
#             dataset_loader.save_sample(img=img, is_show=False, file_name=os.path.join(dataset_img_dir, f"{i}.png"))
#         re_comp_clean_metric = True
    
#     # Init noise
#     noise = torch.randn(
#                 (config.measure_sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
#                 # generator=torch.manual_seed(config.seed),
#             )
#     backdoor_noise = noise + dataset_loader.trigger.unsqueeze(0)
    
#     # Sampling
#     if not os.path.isdir(clean_path) or resample:
#         batch_sampling_save(sample_n=config.measure_sample_n, pipeline=pipeline, path=clean_path, init=noise, max_batch_n=config.eval_max_batch, rng=rng)
#         re_comp_clean_metric = True
#     if not os.path.isdir(backdoor_path) or resample:
#         batch_sampling_save(sample_n=config.measure_sample_n, pipeline=pipeline, path=backdoor_path, init=backdoor_noise,  max_batch_n=config.eval_max_batch, rng=rng)
#         re_comp_backdoor_metric = True
    
#     # Compute Score
#     if re_comp_clean_metric or recomp:
#         fid_sc = float(fid(path=[dataset_img_dir, clean_path], device=config.device_ids[0], num_workers=4))
    
#     if re_comp_backdoor_metric or recomp:
#         device = torch.device(config.device_ids[0])
#         # gen_backdoor_target = torch.from_numpy(backdoor_sample_imgs)
#         # print(f"backdoor_sample_imgs shape: {backdoor_sample_imgs.shape}")
#         gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(device)
        
#         reps = ([len(gen_backdoor_target)] + ([1] * (len(dsl.target.shape))))
#         backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
        
#         logging.info(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
#         mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
#         ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
#     # logging.info(f"[{config.sample_ep}] FID: {fid_sc}, MSE: {mse_sc}, SSIM: {ssim_sc}")
#     logging.info(f"FID: {fid_sc}, MSE: {mse_sc}, SSIM: {ssim_sc}")
    
#     sc = update_score_file(config=config, score_file=score_file, fid_sc=fid_sc, mse_sc=mse_sc, ssim_sc=ssim_sc)
#     # accelerator.log(sc)
#     log_score(config=config, accelerator=accelerator, scores=sc, step=step)

"""With this in end, we can group all together and write our training function. This just wraps the training step we saw in the previous section in a loop, using Accelerate for easy TensorBoard logging, gradient accumulation, mixed precision training and multi-GPUs or TPU training."""

def get_ep_model_path(config, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")

def save_checkpoint(config, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    accelerator.save_state(config.ckpt_path)   
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    pipeline.save_pretrained(config.result_dir)
    
def sampling(config, file_name: Union[int, str], pipeline):
    def gen_samples(init: torch.Tensor, folder: Union[os.PathLike, str]):
        test_dir = os.path.join(config.result_dir, folder)
        os.makedirs(test_dir, exist_ok=True)
        
        pipline_res = pipeline(
            batch_size = 16, # config.eval_sample_n 
            # generator=torch.manual_seed(config.seed),
            init=init,
            output_type=None,
            save_every_step=True
        )
        images = pipline_res.images
        movie = pipline_res.movie
        
        # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
        init_images = [Image.fromarray(image) for image in np.squeeze((movie[0] * 255).round().astype("uint8"))]

        # # Make a grid out of the images
        image_grid = make_grid(images, rows=4, cols=4)
        init_image_grid = make_grid(init_images, rows=4, cols=4)
        
        # clip_opt = "" if config.clip else "_noclip"
        clip_opt = ""
        # # Save the images
        if isinstance(file_name, int):
            image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name:04d}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name:04d}{clip_opt}_sample_t", animate_name=f"{file_name:04d}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        elif isinstance(file_name, str):
            image_grid.save(f"{test_dir}/{file_name}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name}{clip_opt}_sample_t", animate_name=f"{file_name}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        else:
            raise TypeError(f"Argument 'file_name' should be string nor integer.")
    
    with torch.no_grad():
        noise = torch.randn(
                    (16, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                    # generator=torch.manual_seed(config.seed),
                )
        # Sample Clean Samples
        gen_samples(init=noise, folder="samples")
        # Sample Backdoor Samples
        # init = noise + torch.where(dsl.trigger.unsqueeze(0) == -1.0, 0, 1)
        init = noise + dsl.trigger.unsqueeze(0)
        # print(f"Trigger - (max: {torch.max(dsl.trigger)}, min: {torch.min(dsl.trigger)}) | Noise - (max: {torch.max(noise)}, min: {torch.min(noise)}) | Init - (max: {torch.max(init)}, min: {torch.min(init)})")
        gen_samples(init=init, folder="backdoor_samples")

def train_loop(config, accelerator: Accelerator, repo, model: nn.Module, get_pipeline, noise_sched, optimizer: torch.optim, loader, lr_sched, start_epoch: int=0, start_step: int=0):
    try:
        # memlog = MemoryLog('memlog.log')
        cur_step = start_step
        epoch = start_epoch
        
        # Test evaluate
        # memlog.append()
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        # sampling(config, 0, pipeline)
        # memlog.append()

        # Now you train the model
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(loader):
                # memlog.append()
                # clean_images = batch['images']
                clean_images = batch['pixel_values'].to(model.device_ids[0])
                target_images = batch["target"].to(model.device_ids[0])
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device) # 随机采样一个高斯噪声，用于添加到图片当中
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_sched.num_train_timesteps, (bs,), device=clean_images.device).long()
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (bs,), device=clean_images.device).long() # 为每一张图片生成一个时间步
                
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    loss = p_losses_diffuser(noise_sched, model=model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2")
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
                accelerator.log(logs, step=cur_step)
                cur_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)

                # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epoch - 1:
                #     sampling(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
    except:
        logging.error("Training process is interrupted by an error")
        logging.info(traceback.format_exc())
    finally:
        logging.info("Save model and sample images")
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        if accelerator.is_main_process:
            save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # sampling(config, 'final', pipeline)
        return pipeline

if __name__ == "__main__":
    set_random_seeds()
    config = setup()
    
    """## Let's train!

    Let's launch the training (including multi-GPU training) from the notebook using Accelerate's `notebook_launcher` function:
    """
    dsl = get_uncond_data_loader(config=config)
    accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=config, dataset_loader=dsl)
    # train or resume training
    if config.mode == MODE_TRAIN or config.mode == MODE_RESUME:
        pipeline = train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, start_epoch=cur_epoch, start_step=cur_step)

        # if config.mode == MODE_TRAIN_MEASURE and accelerator.is_main_process:
        #     accelerator.free_memory()
        #     accelerator.clear()
        #     measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline)
    elif config.mode == MODE_SAMPLING:
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        sampling(config=config, file_name="final", pipeline=pipeline)
    # elif config.mode == MODE_MEASURE:
    #     # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    #     pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    #     measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline)
    #     # if config.sample_ep != None:
    #     #     sampling(config=config, file_name=int(config.sample_ep), pipeline=pipeline)
    #     # else:
    #     #     sampling(config=config, file_name="final", pipeline=pipeline)
    #     sampling(config=config, file_name="final", pipeline=pipeline)
    else:
        raise NotImplementedError()

    accelerator.end_training()
