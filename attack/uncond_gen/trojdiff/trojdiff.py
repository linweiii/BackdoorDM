import argparse
import os, sys
import json
import traceback
from typing import  Dict
import torchvision.transforms as T
import time
import torch
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from attack.uncond_gen.baddiff_backdoor import BadDiff_Backdoor
from utils.utils import *
from utils.uncond_dataset import DatasetLoader
from utils.load import init_uncond_train, get_uncond_data_loader
from loss import trojdiff_loss, trojdiff_loss_out
from PIL import Image
from accelerate import Accelerator
from tqdm.auto import tqdm
from sample import sample

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'

DEFAULT_BATCH: int = 512
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_TRIGGER: str = BadDiff_Backdoor.TRIGGER_BOX_14
DEFAULT_TARGET: str = BadDiff_Backdoor.TARGET_MICKEY
DEFAULT_GPU = '0, 1'
DEFAULT_CKPT: str = None
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_RESULT: int = 'trojdiff01'
    
def parse_args():
    method_name = 'trojdiff'
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', type=str, help='Project name')
    parser.add_argument('--base_config', type=str, default='./attack/uncond_gen/configs/base_config.yaml')
    parser.add_argument('--bd_config', type=str, default='./attack/uncond_gen/configs/bd_config_fix.yaml')
    parser.add_argument('--mode', '-m', type=str, help='Train or test the model', choices=[MODE_TRAIN, MODE_RESUME])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset', choices=[DatasetLoader.CIFAR10, DatasetLoader.CELEBA_ATTR])
    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler', choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED", "LMSD-SCHED", "SCORE-SDE-VE-SCHED", "EDM-VE-SDE-SCHED", "EDM-VE-ODE-SCHED"])
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    
    # attack
    parser.add_argument('--cond_prob', type=float, default=1.0)
    # parser.add_argument('--gamma', type=float, default=0.6)
    parser.add_argument('--trigger_type', type=str, default='blend')
    
    parser.add_argument('--attack_mode', type=str, default='d2i')
    
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    parser.add_argument('--save_model_epochs', '-sme', type=int, help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--result', '-res', type=str, default='test_trojdiff', help=f"Output file path, default: {DEFAULT_RESULT}")
    
    parser.add_argument('--batch_32', type=int, default=128)
    parser.add_argument('--batch_256', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate_32_scratch', type=float, default=2e-4)
    parser.add_argument('--learning_rate_256_scratch', type=float, default=2e-5)
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    
    parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--data_ckpt_dir', type=str, default='data.ckpt')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--data_ckpt_path', type=str, default=None)
    parser.add_argument('--load_ckpt', type=bool, default=False) # True when resume
    
    parser.add_argument('--seed', type=int, default=35)
    
    args = parser.parse_args()
    args.backdoor_method = method_name
    args = base_args_uncond_v1(args)
    print(args)
    
    return args

def setup():
    config_file: str = "config.json"
    
    args: argparse.Namespace = parse_args()
    args_data: Dict = {}
    
    if args.mode == MODE_RESUME:
        with open(os.path.join('results', args.result, config_file), "r") as f:
            args_data = json.load(f)
        
        for key, value in args_data.items():
            if key == 'ckpt':
                continue
            if value != None:
                setattr(args, key, value)
                
        setattr(args, "result_dir", os.path.join('results', args.result))
        setattr(args, "load_ckpt", True)
        logger = set_logging(f'{args.result_dir}/train_logs/')
    elif args.mode == MODE_TRAIN:
        args.result = args.backdoor_method + '_' + args.ckpt
        setattr(args, "result_dir", os.path.join('results', args.result))
        logger = set_logging(f'{args.result_dir}/train_logs/')
    else:
        raise NotImplementedError
        
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    logger.info(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(args, "device_ids", [int(i) for i in range(len(args.gpu.split(',')))])
        
    # Determine gradient accumulation & Learning Rate
    bs = 0
    if args.dataset in [DatasetLoader.CIFAR10]:
        bs = args.batch_32
        if args.learning_rate == None:
            if args.ckpt == None:
                args.learning_rate = args.learning_rate_32_scratch
            else:
                args.learning_rate = DEFAULT_LEARNING_RATE_32
    elif args.dataset in [DatasetLoader.CELEBA_ATTR]:
        bs = args.batch_256
        if args.learning_rate == None:
            if args.ckpt == None:
                args.learning_rate = args.learning_rate_256_scratch
            else:
                args.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError("Dataset Not supported.")
    
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
    logger.info('Note that trigger, target and poison_rate arguments are useless for TrojDiff. Just ignore them.')
    
    setattr(args, 'batch', bs) # automatically modify batch size according to dataset
    args.gradient_accumulation_steps = int(bs // args.batch)
    
    logger.info(f"MODE: {args.mode}")
    write_json(content=args.__dict__, config=args, file=config_file) # save config
    
    if not hasattr(args, 'ckpt_path') or args.ckpt_path == None:
        args.ckpt_path = os.path.join(args.result_dir, args.ckpt_dir)
        args.data_ckpt_path = os.path.join(args.result_dir, args.data_ckpt_dir)
        os.makedirs(args.ckpt_path, exist_ok=True)
    
    logger.info(f"Argument Final: {args.__dict__}")
    return args, logger


def save_checkpoint(config, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    accelerator.save_state(config.ckpt_path)   
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    pipeline.save_pretrained(config.result_dir)
        
        
def get_target_loader(config, org_size, logger):
    ds_root = os.path.join(config.dataset_path)
    target_dsl = DatasetLoader(root=ds_root, name=config.targetset, label=config.target_label, batch_size=int(config.batch * 0.5), logger=logger).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate).get_targetset(org_size=org_size)
    targetset_loader = target_dsl.get_dataloader()
    
    def cycle(dl):
        while True:
            for data in dl:
                yield data
    
    return cycle(targetset_loader)


def get_target_img(file_path, org_size):
    target_img = Image.open(file_path)
    if target_img.mode == 'RGB' or 'RGBA':
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
    
    
def train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, logger, start_epoch: int=0, start_step: int=0):
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
                logger.info(str(logs))
                accelerator.log(logs, step=cur_step)
                cur_step += 1
            
            if accelerator.is_main_process:
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
    except:
        logging.error("Training process is interrupted by an error")
        logger.info(traceback.format_exc())
    finally:
        logger.info("Save model and sample images")
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        if accelerator.is_main_process:
            save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # sample(config, pipeline, noise_sched, miu)
        return pipeline

def train_loop_out(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, logger, start_epoch: int=0, start_step: int=0):
    try:
        cur_step = start_step
        epoch = start_epoch       
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        
        first_batch = next(iter(loader))
        org_size = first_batch['image'].shape[-1]
        miu = get_target_img(config.miu_path, org_size)
        target_loader = get_target_loader(config, org_size, logger)
        
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
                logger.info(str(logs))
                accelerator.log(logs, step=cur_step)
                cur_step += 1
            
            if accelerator.is_main_process:
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
    except:
        logger.error("Training process is interrupted by an error")
        logger.info(traceback.format_exc())
    finally:
        logger.info("Save model and sample images")
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        if accelerator.is_main_process:
            save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # sample(config, pipeline, noise_sched, miu)
        return pipeline
    
def train_loop_d2i(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, logger, start_epoch: int=0, start_step: int=0):
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
                logger.info(str(logs))
                accelerator.log(logs, step=cur_step)
                cur_step += 1
            
            if accelerator.is_main_process:
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
    except:
        logger.error("Training process is interrupted by an error")
        logger.info(traceback.format_exc())
    finally:
        logger.info("Save model and sample images")
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        if accelerator.is_main_process:
            save_checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # sample(config, pipeline, noise_sched, miu)
        return pipeline

if __name__ == "__main__":
    start = time.time()
    config, logger = setup()
    set_random_seeds(config.seed)
    dsl = get_uncond_data_loader(config, logger)
    accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=config, dataset_loader=dsl)
    if config.mode == MODE_TRAIN or config.mode == MODE_RESUME:
        if config.attack_mode == 'd2d-in':
            pipeline = train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, logger, start_epoch=cur_epoch, start_step=cur_step)
        elif config.attack_mode == 'd2d-out':
            pipeline = train_loop_out(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, logger, start_epoch=cur_epoch, start_step=cur_step)
        elif config.attack_mode == 'd2i':
            pipeline = train_loop_d2i(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, logger, start_epoch=cur_epoch, start_step=cur_step)
        else:
            raise NotImplementedError()
            
    accelerator.end_training()
    end = time.time()
    logger.info(f'Total time: {end-start}s')