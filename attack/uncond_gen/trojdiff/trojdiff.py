import argparse
import os, sys
import json
import traceback
from typing import Callable, Dict, List, Tuple, Union
import torchvision.transforms as T

import torch
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from attack.uncond_gen.baddiff_backdoor import BadDiff_Backdoor
from utils.utils import *
from utils.uncond_dataset import DatasetLoader, ImagePathDataset
from utils.load import init_uncond_train, get_uncond_data_loader
from loss import trojdiff_loss, trojdiff_loss_out
from PIL import Image
from torch import nn
from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo, push_to_hub
from tqdm.auto import tqdm
from sample import sample

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'

DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_EVAL_MAX_BATCH: int = 256
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
    with open('./attack/uncond_gen/configs/trojdiff.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
        return config
    
def parse_args():
    args_config = load_config_from_yaml()
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', type=str, help='Project name')
    parser.add_argument('--mode', '-m', type=str, help='Train or test the model', choices=[MODE_TRAIN, MODE_RESUME, MODE_SAMPLING, MODE_MEASURE, MODE_TRAIN_MEASURE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset', choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.CELEBA_HQ_LATENT_PR05, DatasetLoader.CELEBA_HQ_LATENT])
    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler', choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED", "LMSD-SCHED", "SCORE-SDE-VE-SCHED", "EDM-VE-SDE-SCHED", "EDM-VE-ODE-SCHED"])
    # parser.add_argument('--ddim_eta', '-det', type=float, help=f'Randomness hyperparameter \eta of DDIM, range: [0, 1], default: {DEFAULT_DDIM_ETA}')
    # parser.add_argument('--infer_steps', '-is', type=int, help='Number of inference steps')
    # parser.add_argument('--infer_start', '-ist', type=float, help='Inference start timestep')
    # parser.add_argument('--inpaint_mul', '-im', type=float, help='Inpainting initial sampler multiplier')
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--eval_max_batch', '-eb', type=int, help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float, default=0, help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    # attack
    parser.add_argument('--cond_prob', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--miu_path', type=str, default='./attack/uncond_gen/static/hello_kitty.png')
    parser.add_argument('--trigger_type', type=str, default='blend')
    parser.add_argument('--patch_size', type=int, default=3)
    
    parser.add_argument('--attack_mode', type=str, default='d2d-out')
    parser.add_argument('--targetset', type=str, default="MNIST")
    parser.add_argument('--target_img', type=str, default='./attack/uncond_gen/static/mickey.png')
    
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
    
    # sample
    parser.add_argument('--img_num_test', type=int, default=16) 
    parser.add_argument('--infer_steps', '-is', type=int, default=1000)
    parser.add_argument("--sample_type",type=str, default="ddpm_noisy",help="sampling approach (ddim_noisy or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)")
    
    args = parser.parse_args()
    for key in vars(args):
        if getattr(args, key) is not None:
            args_config[key] = getattr(args, key)
    
    final_args = argparse.Namespace(**args_config)
    print(final_args)
    
    return final_args

def setup():
    set_random_seeds()
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
    
    if args.attack_mode == 'd2d-out':
        if not hasattr(args, 'targetset') or args.targetset == None:
            logging.error('Must specify targetset for d2d-out mode!')
            raise ValueError('Must specify targetset for d2d-out mode!')
    elif args.attack_mode == 'd2i':
        if not hasattr(args, 'target_img') or args.target_img == None:
            logging.error('Must specify target_img for d2d-out mode!')
            raise ValueError('Must specify target_img for d2d-out mode!')
    else:
        if args.attack_mode != 'd2d-in':
            raise NotImplementedError()
    
    if args.trigger_type == 'patch':
        if not hasattr(args, 'patch_size') or args.patch_size == None:
            logging.error('Must specify patch_size for patch mode!')
            raise ValueError('Must specify patch_size for patch mode!')
    else:
        if args.trigger_type != 'blend':
            raise NotImplementedError()
        
    setattr(args, 'trigger', DEFAULT_TRIGGER)
    setattr(args, 'target', DEFAULT_TARGET)
    logging.info('Note that trigger, target and poison_rate arguments are useless for TrojDiff. Just ignore them.')
    
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


def get_ep_model_path(config, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")

def save_checkpoint(config, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    accelerator.save_state(config.ckpt_path)   
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    pipeline.save_pretrained(config.result_dir)
        
        
def get_target_loader(config, org_size):
    ds_root = os.path.join(config.dataset_path)
    target_dsl = DatasetLoader(root=ds_root, name=config.targetset, batch_size=int(config.batch * 0.5)).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate).get_targetset(org_size=org_size)
    targetset_loader = target_dsl.get_dataloader()
    
    def cycle(dl):
        while True:
            for data in dl:
                yield data
    
    return cycle(targetset_loader)


def get_target_img(file_path, org_size):
    target_img = Image.open(file_path)
    if target_img.mode == 'RGB':
        channel_trans = T.Lambda(lambda x: x.convert("RGB"))
    elif target_img.mode == 'L':
        channel_trans = T.Grayscale(num_output_channels=1)
    else:
        logging.error('Not support this target image.')
        raise NotImplementedError('Not support this target image.')
    transform = T.Compose([channel_trans,
                T.Resize([org_size, org_size]), 
                T.ToTensor(),
                T.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1.0, vmax_out=1.0, x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ])
    target_img = transform(target_img)
    
    return target_img
    
    
def train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, start_epoch: int=0, start_step: int=0):
    try:
        cur_step = start_step
        epoch = start_epoch       
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        first_batch = next(iter(loader))
        org_size = first_batch['image'].shape[-1]
        miu = get_target_img(config.miu_path, org_size)
        
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in enumerate(loader):
                clean_images = batch['image'].to(model.device_ids[0])
                labels = batch['label'].to(model.device_ids[0])
                n = clean_images.shape[0]
                # antithetic sampling
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (n // 2 + 1,), device=clean_images.device).long()
                timesteps = torch.cat([timesteps, noise_sched.config.num_train_timesteps - timesteps - 1], dim=0)[:n]
                
                with accelerator.accumulate(model):
                    loss = trojdiff_loss(config, noise_sched, model, clean_images, labels, timesteps, miu, config.target_label, config.gamma, config.cond_prob)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                    
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=cur_step)
                cur_step += 1
            
            if accelerator.is_main_process:
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
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
            # sample(config, pipeline, noise_sched, miu)
        return pipeline

def train_loop_out(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, start_epoch: int=0, start_step: int=0):
    try:
        cur_step = start_step
        epoch = start_epoch       
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        
        first_batch = next(iter(loader))
        org_size = first_batch['image'].shape[-1]
        miu = get_target_img(config.miu_path, org_size)
        target_loader = get_target_loader(config, org_size)
        
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in enumerate(loader):
                x = batch['image'].to(model.device_ids[0])
                y = batch['label'].to(model.device_ids[0])
                x_tar = next(iter(target_loader))
                x_tar = x_tar['image'].to(model.device_ids[0])
                y_tar = torch.ones(x_tar.shape[0]) * 1000
                y_tar = y_tar.to(model.device_ids[0])
                clean_images = torch.cat([x, x_tar], dim=0).to(model.device_ids[0])
                labels = torch.cat([y, y_tar], dim=0).to(model.device_ids[0])
                n = clean_images.shape[0]
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (n // 2 + 1,), device=clean_images.device).long()
                timesteps = torch.cat([timesteps, noise_sched.config.num_train_timesteps - timesteps - 1], dim=0)[:n]
                
                with accelerator.accumulate(model):
                    loss = trojdiff_loss_out(config, noise_sched, model, clean_images, labels, timesteps, miu, config.gamma, config.cond_prob)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                    
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=cur_step)
                cur_step += 1
            
            if accelerator.is_main_process:
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
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
            # sample(config, pipeline, noise_sched, miu)
        return pipeline
    
def train_loop_d2i(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, start_epoch: int=0, start_step: int=0):
    try:
        cur_step = start_step
        epoch = start_epoch       
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        
        first_batch = next(iter(loader))
        org_size = first_batch['image'].shape[-1]
        miu = get_target_img(config.miu_path, org_size)
        target_img = get_target_img(config.target_img, org_size)
        
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in enumerate(loader):
                x = batch['image'].to(model.device_ids[0])
                y = batch['label'].to(model.device_ids[0])
                bs = x.shape[0]
                target_bs = int(bs * 0.1)
                x_tar = torch.stack([target_img] * target_bs).to(model.device_ids[0])
                y_tar = torch.ones(target_bs) * 1000
                y_tar = y_tar.to(model.device_ids[0])
                clean_images = torch.cat([x, x_tar], dim=0).to(model.device_ids[0])
                labels = torch.cat([y, y_tar], dim=0).to(model.device_ids[0])
                n = clean_images.size(0)
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (n // 2 + 1,), device=clean_images.device).long()
                timesteps = torch.cat([timesteps, noise_sched.config.num_train_timesteps - timesteps - 1], dim=0)[:n]
                
                with accelerator.accumulate(model):
                    loss = trojdiff_loss_out(config, noise_sched, model, clean_images, labels, timesteps, miu, config.gamma, config.cond_prob)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                    
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=cur_step)
                cur_step += 1
            
            if accelerator.is_main_process:
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
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
            sample(config, pipeline, noise_sched, miu)
        return pipeline

if __name__ == "__main__":
    config = setup()
    dsl = get_uncond_data_loader(config = config)
    accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=config, dataset_loader=dsl)
    if config.mode == MODE_TRAIN or config.mode == MODE_RESUME:
        if config.attack_mode == 'd2d-in':
            pipeline = train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, start_epoch=cur_epoch, start_step=cur_step)
        elif config.attack_mode == 'd2d-out':
            pipeline = train_loop_out(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, start_epoch=cur_epoch, start_step=cur_step)
        elif config.attack_mode == 'd2i':
            pipeline = train_loop_d2i(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, start_epoch=cur_epoch, start_step=cur_step)
        else:
            raise NotImplementedError()
            
    accelerator.end_training()