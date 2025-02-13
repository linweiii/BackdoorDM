import argparse
import os, sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from collections import defaultdict
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from piq import LPIPS

import torch
from torch import optim
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDIMPipeline, DDIMScheduler
from utils.uncond_model import DiffuserModelSched, DiffuserModelSched_SDE
from utils.load import init_uncond_train, get_uncond_data_loader
from utils.utils import *
from attack.uncond_gen.bad_diffusion.loss import p_losses_diffuser
from attack.uncond_gen.trojdiff.loss import noise_estimation_loss
from attack.uncond_gen.villan_diffusion.loss import LossFn


def load_config_from_yaml():
    with open('./defense/model_level/configs/elijah.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
        return config
    

def trigger_loss(noise, output):
    noise = noise.mean(0)
    output = output.mean(0)
    # save_image(noise.unsqueeze(0)*0.5+0.5, './tmp_noise.png')
    # save_image(output.unsqueeze(0)*0.5+0.5, './tmp_output.png')
    # print(noise.shape, output.shape)
    loss = torch.nn.functional.l1_loss(noise, output)
    return loss

def compute_uniformity(images):
    if images.shape[-1] == 3:
        # last channle is rgb
        images = rearrange(images, 'b h w c -> b c h w')

    images1 = repeat(images, 'b c h w -> (b tile) c h w', tile=len(images))
    images2 = repeat(images, 'b c h w -> (tile b) c h w', tile=len(images))

    percept = LPIPS(replace_pooling=True, reduction="none")
    loss = percept(images1, images2).view(len(images), len(images))  # .mean()
    loss = torch.sort(loss, dim=1)[0]
    skip_cnt = 4
    loss = loss[:, skip_cnt:-skip_cnt]
    loss = loss.mean(dim=1)
    loss = torch.sort(loss)[0]
    loss = loss[skip_cnt:-skip_cnt].mean()

    return loss.item()

from torchmetrics.image import TotalVariation
def compute_tvloss(images):
    if images.shape[-1] == 3:
        # last channle is rgb
        images = rearrange(images, 'b h w c -> b c h w')

    tv = TotalVariation(reduction='mean').cuda()

    return tv(images).item()

def format_ckpt_dir(ckpt):
    return ckpt.replace('/', '_')

@torch.no_grad()
def sample_with_trigger(args, trigger, file_name, logger, R_coef_T, recomp=False, use_ddim=False, save_res_dict=False):
    test_dir = os.path.join(format_ckpt_dir(args.ckpt), 'defenses', args.defense_result,'generated_img_with_trigger')
    os.makedirs(test_dir, exist_ok=True)
    generated_img_ptfile = os.path.join(test_dir, f'{file_name}.pt')
    result_dict = {}
    if not os.path.isfile(generated_img_ptfile) or recomp:
        if use_ddim:
            model, noise_sched, get_pipeline = DiffuserModelSched.new_get_pretrained(ckpt=args.ckpt, noise_sched_type=DiffuserModelSched.DDIM_SCHED)
            unet = model.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = get_pipeline(unet=unet, scheduler=noise_sched)
        else:
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=args.ckpt)
            unet = model.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = DDPMPipeline(unet=unet, scheduler=noise_sched)
            
        def gen_samples(init):

            if use_ddim:
                pipeline_res = pipeline(
                    batch_size = 16,
                    init=init,
                    output_type=None,
                    num_inference_steps=50
                )
            else:
                pipeline_res = pipeline(
                    batch_size = 16,
                    init=init,
                    output_type=None,
                    return_full_mov=False
                )   
            images = pipeline_res.images
            movie = pipeline_res.movie
            if args.compute_tvloss:
                loss = compute_tvloss(torch.from_numpy(images).cuda())
            else:
                loss = compute_uniformity(torch.from_numpy(images).cuda())
            result_dict[R_coef_T] = loss
            torch.save(torch.from_numpy(images), generated_img_ptfile)
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            init_images = [Image.fromarray(image) for image in np.squeeze((movie[0] * 255).round().astype("uint8"))]
        
        with torch.no_grad():
            noise = torch.randn([16, ] + noise_shape)
            init = noise + trigger
            gen_samples(init=init)
    elif R_coef_T not in result_dict:
        images = torch.load(generated_img_ptfile)
        if args.compute_tvloss:
            logger.info('Use TV Loss')
            loss = compute_tvloss(images.cuda())
        else:
            logger.info('Use Uniformity')
            loss = compute_uniformity(images.cuda())
        result_dict[R_coef_T] = loss
    else:
        pass
    logger.info(f'{args.ckpt}@{R_coef_T}: {result_dict[R_coef_T]}')
    if save_res_dict:
        if args.compute_tvloss:
            res_name = 'res_dict_tvloss.pt'
        else:
            res_name = 'res_dict.pt'
        res_path = os.path.join(args.ckpt, 'defenses', args.defense_result, res_name)
        torch.save(result_dict, res_path)
            

def trigger_inversion(args, logger, detect=False):
    R_coef_T = 0.5
    trigger_filename = args.ckpt + f'/defenses/{args.defense_result}/inverted_trigger/trigger_{R_coef_T}.pt'
    if not os.path.isdir(os.path.dirname(trigger_filename)):
        os.makedirs(os.path.dirname(trigger_filename))
    
    if not os.path.isfile(trigger_filename):
        model, noise_sched, _ = DiffuserModelSched.get_pretrained(ckpt=args.ckpt)
        unet = model.cuda()
        noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
        if noise_shape[-1] == 256:
            bs = 20
        elif noise_shape[-1] == 128:
            bs = 50
        else:
            bs = 100
        noise = torch.randn([bs, ] + noise_shape).cuda()
        T = noise_sched.num_train_timesteps - 1
        logger.info('#####Start trigger inversion#####')
        logger.info(f'R_coef_T: {R_coef_T}')
        
        trigger = -torch.rand([1, ] + noise_shape).cuda()
        trigger.requires_grad_(True)
        optimizer = optim.Adam([trigger, ], lr=0.1)
        num_epochs = 100
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            unet.zero_grad()
            
            trigger_noise = noise + trigger
            model_output = unet(trigger_noise, T).sample
            loss = trigger_loss(trigger * R_coef_T, model_output)
            loss.backward()
            optimizer.step()
            logger.info(f'{epoch} loss: {loss.item()}, R_coef_T: {R_coef_T if isinstance(R_coef_T, float) else R_coef_T.item()}')
            
        if not isinstance(R_coef_T, float):
            R_coef_T = R_coef_T.item()
        torch.save(trigger.cpu(), trigger_filename)
        save_tensor_img(trigger.cpu(), args.ckpt + f'/defenses/{args.defense_result}/inverted_trigger/trigger_{R_coef_T}_img.png')
    else:
        trigger = torch.load(trigger_filename, map_location='cpu')
    
    filename = 'inverted_{R_coef_T}'
    if detect:
        sample_with_trigger(args.ckpt, trigger.cpu(), filename, logger, R_coef_T, save_res_dict=True)
    
    return trigger_filename


def trigger_inversion_sde(args, logger, detect=False):
    R_coef_T = 0.5
    trigger_filename = args.ckpt + f'/defenses/{args.defense_result}/inverted_trigger/trigger_{R_coef_T}.pt'
    if not os.path.isdir(os.path.dirname(trigger_filename)):
        os.makedirs(os.path.dirname(trigger_filename))
    
    if not os.path.isfile(trigger_filename):
        model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=args.ckpt, clip_sample=False, noise_sched_type=args.sched, sde_type=args.sde_type)
        unet = model.cuda()
        if vae is not None:
            vae = vae.cuda()
        noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
        print(args.sched)
        print(args.sde_type)
        pipeline = get_pipeline(unet=unet, vae=vae, scheduler=noise_sched)
        
        if noise_shape[-1] == 256:
            bs = 20
        elif noise_shape[-1] == 128:
            bs = 50
        else:
            bs = 100
        if args.sde_type == DiffuserModelSched_SDE.SDE_LDM:
            bs = 40
        
        noise = torch.randn([bs, ] + noise_shape).cuda()
        if args.sde_type == DiffuserModelSched_SDE.SDE_VE:
            T = 3.8 * 100
            trigger = torch.rand([1, ] + noise_shape).cuda()
            num_epochs = 100
        elif args.sde_type == DiffuserModelSched_SDE.SDE_LDM:
            T = noise_sched.num_train_timesteps - 1 # 999
            trigger = torch.rand([1, ] + noise_shape).cuda() * 2 - 1
            num_epochs = 10
            del vae
            del pipeline
        else:
            T = noise_sched.num_train_timesteps - 1
            trigger = torch.rand([1, ] + noise_shape).cuda() * 2 - 1
            num_epochs = 10
            if vae is not None:
                trigger = pipeline.encode(trigger)
                del vae
            del pipeline
            
        logger.info('#####Start trigger inversion#####')
        logger.info('R_coef_T', R_coef_T)
        
        logger.info('trigger stat:', trigger.min(), trigger.max())
        trigger.requires_grad_(True)
        optimizer = optim.Adam([trigger, ], lr=0.1)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            unet.zero_grad()
            if args.sde_type == DiffuserModelSched_SDE.SDE_VE:
                trigger_noise = 380 * (noise + trigger)
                model_output = unet(trigger_noise, T).sample
                loss = trigger_loss(trigger, model_output / (-R_coef_T/380))
            else:
                trigger_noise = noise + trigger
                model_output = unet(trigger_noise, T).sample
                loss = trigger_loss(2 * trigger * R_coef_T, model_output)
                
            loss.backward()
            optimizer.step()
            logger.info(f'{epoch} loss: {loss.item()}, R_coef_T: {R_coef_T if isinstance(R_coef_T, float) else R_coef_T.item()}')
            
        if not isinstance(R_coef_T, float):
            R_coef_T = R_coef_T.item()
        with torch.no_grad():
            torch.save(trigger.cpu(), trigger_filename)
            save_tensor_img(trigger.cpu(), args.ckpt + f'/defenses/{args.defense_result}/inverted_trigger/trigger_{R_coef_T}_img.png')
    else:
        trigger = torch.load(trigger_filename, map_location='cpu')
    
    filename = 'inverted_{R_coef_T}'
    if detect:
        sample_with_trigger(args.ckpt, trigger.cpu(), filename, logger, R_coef_T, save_res_dict=True)
    
    return trigger_filename

    
@torch.no_grad()
def get_frozen_model(args):
    assert args.ckpt != None
    
    model, noise_sched, _ = DiffuserModelSched.get_pretrained(ckpt=args.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    return model

@torch.no_grad()
def get_frozen_model_sde(args):
    assert args.ckpt != None
    
    model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=args.ckpt, clip_sample=False, noise_sched_type=args.sched, sde_type=args.sde_type)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def deshift_loss(model, noise, trigger, T, frozen_model):
    benign_prediction = model(noise, T, return_dict=False)[0]
    with torch.no_grad():
        frozen_benign_prediction = frozen_model(noise, T, return_dict=False)[0]
    backdoor_prediction = model(noise+trigger, T, return_dict=False)[0]

    # loss1 = torch.nn.functional.mse_loss(backdoor_prediction, benign_prediction)
    # loss1 = torch.nn.functional.mse_loss(backdoor_prediction, benign_prediction.detach())
    loss1 = torch.nn.functional.mse_loss(backdoor_prediction, frozen_benign_prediction)

    # this loss can reduce the trigger effect in 20 updates with 128 batch size, very fast!!
    # only this loss makes the fine-tuned model outputs trigger when intput noises are benign
    # therefore, we can frozen original backdoor model M0, and add another loss s.t. M0(noise) = M'(noise)
    # with torch.no_grad():
    #     frozen_benign_prediction = frozen_model(noise, T, return_dict=False)[0]
    loss2 = torch.nn.functional.mse_loss(benign_prediction, frozen_benign_prediction)
    # loss2 = 0
    return loss1, loss2

def deshift_loss_troj(model, noise, trigger, T, frozen_model):
    benign_prediction = model(noise, T, return_dict=False)[0]
    with torch.no_grad():
        frozen_benign_prediction = frozen_model(noise, T, return_dict=False)[0]
    backdoor_prediction = model(noise+trigger, T, return_dict=False)[0]

    # loss1 = torch.nn.functional.mse_loss(backdoor_prediction, frozen_benign_prediction)
    # loss2 = torch.nn.functional.mse_loss(benign_prediction, frozen_benign_prediction)
    loss1 = (frozen_benign_prediction - backdoor_prediction).square().sum(dim=(1, 2, 3)).mean(dim=0)
    loss2 = (frozen_benign_prediction - benign_prediction).square().sum(dim=(1, 2, 3)).mean(dim=0)
    
    return loss1, loss2


def deshift_loss2(model, frozen_model, noise_sched, t_minus_1, benign_x_t_minus_1, backdoor_x_t_minus_1):
    # constrain the t-1->t-2 step
    benign_prediction = model(benign_x_t_minus_1, t_minus_1, return_dict=False)[0]
    with torch.no_grad():
        frozen_benign_prediction = frozen_model(benign_x_t_minus_1, t_minus_1, return_dict=False)[0]
    backdoor_prediction = model(backdoor_x_t_minus_1, t_minus_1, return_dict=False)[0]
    loss1 = torch.nn.functional.mse_loss(backdoor_prediction, frozen_benign_prediction)
    loss2 = torch.nn.functional.mse_loss(benign_prediction, frozen_benign_prediction)
    
    return loss1, loss2

def deshift_loss_sde(args, model, noise, trigger, T, frozen_model):
    # TODO: update the code for score-based models.
    if args.sde_type in [DiffuserModelSched_SDE.SDE_VP, DiffuserModelSched_SDE.SDE_LDM]:
        benign_prediction = model(noise, T, return_dict=False)[0]
        with torch.no_grad():
            frozen_benign_prediction = frozen_model(noise, T, return_dict=False)[0]
        backdoor_prediction = model(noise+trigger, T, return_dict=False)[0]

        loss1 = torch.nn.functional.mse_loss(backdoor_prediction, frozen_benign_prediction)
        loss2 = torch.nn.functional.mse_loss(benign_prediction, frozen_benign_prediction)
    elif args.sde_type == DiffuserModelSched_SDE.SDE_VE:
        noise = 380 * noise
        trigger = 380 * trigger
        T = 380.
        benign_prediction = model(noise, T, return_dict=False)[0]
        with torch.no_grad():
            frozen_benign_prediction = frozen_model(noise, T, return_dict=False)[0]
        backdoor_prediction = model(noise+trigger, T, return_dict=False)[0]

        loss1 = torch.nn.functional.mse_loss(backdoor_prediction, frozen_benign_prediction)
        loss2 = torch.nn.functional.mse_loss(benign_prediction, frozen_benign_prediction)
    else:
        raise NotImplementedError(f"sde_type: {args.sde_type} isn't implemented")

    return loss1, loss2

def save_checkpoint(config, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    ckpt_path = os.path.join(config.result_dir, 'defenses', config.defense_result, 'ckpt')
    data_ckpt_path = os.path.join(config.result_dir, 'defenses', config.defense_result, 'data.ckpt')
    defense_result = os.path.join(config.result_dir, 'defenses', config.defense_result)
    accelerator.save_state(ckpt_path)   
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, data_ckpt_path)
    pipeline.save_pretrained(defense_result) 
    
    
def remove_baddiffusion(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, inverted_trigger, logger, start_epoch=0, start_step=0):
    frozen_model = get_frozen_model(args).to(model.device_ids[0])
    deshift_iters = 0
    cur_step = start_step
    epoch = start_epoch
    
    for epoch in range(int(start_epoch), int(args.epoch)):
        progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        logger.info(f'deshift_iters: {deshift_iters}')
        for step, batch in enumerate(loader):
            clean_images = batch['pixel_values'].to(model.device_ids[0])
            target_images = batch["target"].to(model.device_ids[0])
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(0, noise_sched.num_train_timesteps, (bs,), device=clean_images.device).long()

            with accelerator.accumulate(model):
                    # Predict the noise residual
                loss0 = p_losses_diffuser(noise_sched, model=model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2")
                
                deshift_iters += 1
                loss1, loss2 = deshift_loss(model, noise, inverted_trigger, noise_sched.num_train_timesteps-1, frozen_model)
                loss = loss0 + loss1 + loss2
                accelerator.backward(loss)
                    
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            # logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
            logs = {"loss0": loss0.detach().item(), "loss1": loss1 and loss1.detach().item(), "loss2": loss2 and loss2.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=cur_step)
            logger.info(str(logs))
            cur_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.epoch - 1:
                save_checkpoint(config=args, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
    logger.info("Save repaired model")
    pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    if accelerator.is_main_process:
        save_checkpoint(config=args, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
        
    return pipeline


def remove_trojdiff(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, inverted_trigger, logger, start_epoch=0, start_step=0):
    frozen_model = get_frozen_model(args).to(model.device_ids[0])
    deshift_iters = 0
    cur_step = start_step
    epoch = start_epoch
    for epoch in range(int(start_epoch), int(args.epoch)):
        progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        logger.info(f'deshift_iters: {deshift_iters}')
        for step, batch in enumerate(loader):
            clean_images = batch['image'].to(model.device_ids[0])
            target_images = batch["target"].to(model.device_ids[0])
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            n = clean_images.size(0)

            timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (n // 2 + 1,), device=clean_images.device).long()
            timesteps = torch.cat([timesteps, noise_sched.config.num_train_timesteps - timesteps - 1], dim=0)[:n]

            with accelerator.accumulate(model):
                # Predict the noise residual
                loss0 = noise_estimation_loss(model, noise_sched, clean_images, timesteps, noise)
                # loss0 = p_losses_diffuser(noise_sched, model=model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2")
                deshift_iters += 1
                loss1, loss2 = deshift_loss_troj(model, noise, inverted_trigger, noise_sched.num_train_timesteps-1, frozen_model)
                loss = loss0 + loss1 + loss2
                accelerator.backward(loss)
                    
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            # logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
            logs = {"loss0": loss0.detach().item(), "loss1": loss1 and loss1.detach().item(), "loss2": loss2 and loss2.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=cur_step)
            logger.info(str(logs))
            cur_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.epoch - 1:
                save_checkpoint(config=args, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
    logger.info("Save repaired model")
    pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    if accelerator.is_main_process:
        save_checkpoint(config=args, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
        
    return pipeline

def remove_villandiffusion(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, loader, lr_sched, logger, vae=None, start_epoch=0, start_step=0, inverted_trigger=None):
    weight_dtype: str = None
    scaling_factor: float = 1.0
    model.requires_grad_(True)
    if vae != None:
        vae.requires_grad_(False)
    frozen_model = get_frozen_model_sde(args).to(model.device_ids[0])
    cur_step = start_step
    epoch = start_epoch
    loss_fn = LossFn(noise_sched=noise_sched, sde_type=args.sde_type, loss_type="l2", psi=args.psi, solver_type=args.solver_type, vp_scale=args.vp_scale, ve_scale=args.ve_scale)
    if vae != None:
        vae = accelerator.unwrap_model(vae)       
    pipeline = get_pipeline(accelerator.unwrap_model(model), vae, noise_sched)
    deshift_iters = 0
    for epoch in range(int(start_epoch), int(args.epoch)):
        progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        logger.info(f'deshift_iters: {deshift_iters}')
        for step, batch in enumerate(loader):
            clean_images = batch['pixel_values']
            noise = torch.randn(clean_images.shape).to(model.device_ids[0])
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            with accelerator.accumulate(model):
                loss0 = loss_fn.p_loss_by_keys(batch=batch, model=model, vae=None, target_latent_key="target", poison_latent_key="pixel_values", timesteps=timesteps, noise=None, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
                deshift_iters += 1
                loss1, loss2 = deshift_loss_sde(args, model, noise, inverted_trigger, noise_sched.config.num_train_timesteps-1, frozen_model)
                loss = loss0 + loss1 + loss2
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()
            progress_bar.update(1)    
            logs = {"loss0": loss0.detach().item(), "loss1": loss1 and loss1.detach().item(), "loss2": loss2 and loss2.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=cur_step)
            logger.info(str(logs))
            cur_step += 1   
        if accelerator.is_main_process:
            if vae != None:
                vae = accelerator.unwrap_model(vae)       
            pipeline = get_pipeline(accelerator.unwrap_model(model), vae, noise_sched)
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.epoch - 1:
                save_checkpoint(config=args, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
    logger.info("Save repaired model")
    if vae != None:
        vae = accelerator.unwrap_model(vae)       
    pipeline = get_pipeline(accelerator.unwrap_model(model), vae, noise_sched)
    if accelerator.is_main_process:
        save_checkpoint(config=args, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")

       
def mitigate(removal=False):
    
    args_config = load_config_from_yaml()
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_tvloss', action='store_true', help='compute tv loss instead of uniformity')
    parser.add_argument('--backdoor_method', default='invi_backdoor')
    parser.add_argument('--backdoored_model_path', default='./results/invi_backdoor_DDPM-CIFAR10-32', help='checkpoint')
    
    parser.add_argument('--epoch', default=11)
    parser.add_argument('--clean_rate', default=0.1) # 50 20 11
    parser.add_argument('--save_model_epoch', default=1)
    parser.add_argument('--defense_result', default='elijah')
    parser.add_argument('--seed', type=int, default=35)
    cmd_args = parser.parse_args()
    if cmd_args.backdoor_method == 'trojdiff':
        cmd_args.epoch = 500
    for key in vars(cmd_args):
        if getattr(cmd_args, key) is not None:
            args_config[key] = getattr(cmd_args, key)
    final_args = argparse.Namespace(**args_config)
    setattr(final_args, 'ckpt', cmd_args.backdoored_model_path)
    logger = set_logging(f'{final_args.ckpt}/defenses/{final_args.defense_result}/logs/')
    args = base_args_uncond_defense(final_args)
    set_random_seeds(args.seed)
    args.poison_rate = 0
    dsl = get_uncond_data_loader(args, logger, 'FLEX')
    if hasattr(args, 'sde_type'):
        accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
        if args.sde_type == DiffuserModelSched_SDE.SDE_VP:
            args.epoch = 50
            args.learning_rate = 2e-5
        elif args.sde_type == DiffuserModelSched_SDE.SDE_VE:
            args.epoch = 11
            args.learning_rate = 2e-5
        elif args.sde_type == DiffuserModelSched_SDE.SDE_LDM:
            args.epoch = 20
            args.learning_rate = 2e-4
    else:
        accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)

    if args.backdoor_method == 'villandiffusion':
        trigger_filename = trigger_inversion_sde(args, logger)
    else:
        trigger_filename = trigger_inversion(args, logger)
    
    if removal:
        inverted_trigger = torch.load(trigger_filename, map_location='cpu').to(model.device_ids[0])
        logger.info(f"Using inverted trigger from {trigger_filename}")
        if args.backdoor_method in ['baddiffusion', 'invi_backdoor']:
            pipeline = remove_baddiffusion(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, inverted_trigger, logger)
        elif args.backdoor_method == 'trojdiff':
            pipeline = remove_trojdiff(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, inverted_trigger, logger)
        elif args.backdoor_method == 'villandiffusion':
            pipeline = remove_villandiffusion(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, logger, vae=vae, inverted_trigger=inverted_trigger)
        
if __name__ == '__main__':
    mitigate(True)
    
    
    
    