from diffusers import UNet2DModel, VQModel, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, PNDMScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, ScoreSdeVeScheduler, KarrasVeScheduler, DiffusionPipeline, DDPMPipeline, DDIMPipeline, PNDMPipeline, ScoreSdeVePipeline, LDMPipeline, KarrasVePipeline
from torch import nn
import torch
from functools import partial
from typing import List, Optional, Union
import numpy as np
import logging

class DiffuserModelSched():
    CLIP_SAMPLE_DEFAULT = False
    MODEL_DEFAULT = "DEFAULT"
    DDPM_CIFAR10_DEFAULT = "DDPM-CIFAR10-DEFAULT"
    DDPM_CELEBA_HQ_DEFAULT = "DDPM-CELEBA-HQ-DEFAULT"
    DDPM_CHURCH_DEFAULT = "DDPM-CHURCH-DEFAULT"
    DDPM_BEDROOM_DEFAULT = "DDPM-BEDROOM-DEFAULT"
    LDM_CELEBA_HQ_DEFAULT = "LDM-CELEBA-HQ-DEFAULT"
    
    DDPM_CIFAR10_32 = "DDPM-CIFAR10-32"
    DDPM_CELEBA_HQ_256 = "DDPM-CELEBA-HQ-256"
    DDPM_CHURCH_256 = "DDPM-CHURCH-256"
    DDPM_BEDROOM_256 = "DDPM-BEDROOM-256"
    LDM_CELEBA_HQ_256 = "LDM-CELEBA-HQ-256"

    DDPM_SCHED = "DDPM-SCHED"
    DDIM_SCHED = "DDIM-SCHED"
    DPM_SOLVER_PP_O1_SCHED = "DPM_SOLVER_PP_O1-SCHED"
    DPM_SOLVER_O1_SCHED = "DPM_SOLVER_O1-SCHED"
    DPM_SOLVER_PP_O2_SCHED = "DPM_SOLVER_PP_O2-SCHED"
    DPM_SOLVER_O2_SCHED = "DPM_SOLVER_O2-SCHED"
    DPM_SOLVER_PP_O3_SCHED = "DPM_SOLVER_PP_O3-SCHED"
    DPM_SOLVER_O3_SCHED = "DPM_SOLVER_O3-SCHED"
    UNIPC_SCHED = "UNIPC-SCHED"
    PNDM_SCHED = "PNDM-SCHED"
    DEIS_SCHED = "DEIS-SCHED"
    HEUN_SCHED = "HEUN-SCHED"
    LMSD_SCHED = "LMSD-SCHED"
    LDM_SCHED = "LDM-SCHED"
    SCORE_SDE_VE_SCHED = "SCORE-SDE-VE-SCHED"
    EDM_VE_SCHED = "EDM-VE-SCHED"
    EDM_VE_ODE_SCHED = "EDM-VE-ODE-SCHED"
    EDM_VE_SDE_SCHED = "EDM-VE-SDE-SCHED"

    @staticmethod
    def get_sample_clip(clip_sample: bool, clip_sample_default: bool):
        if clip_sample is not None:
            return clip_sample
        return clip_sample_default
    
    @staticmethod
    def __get_pipeline_generator(unet, scheduler, pipeline):
        def get_pipeline(unet, scheduler):
            return pipeline(unet, scheduler)
        return get_pipeline

    @staticmethod
    def __get_model_sched(ckpt_id: str, clip_sample: bool, noise_sched_type: str=None):
        # Clip option
        clip_sample_used = DiffuserModelSched.get_sample_clip(clip_sample=clip_sample, clip_sample_default=DiffuserModelSched.CLIP_SAMPLE_DEFAULT)
        # Pipeline
        pipline: DDPMPipeline = DDPMPipeline.from_pretrained(ckpt_id)
        
        model: UNet2DModel = pipline.unet
        # noise_sched = pipline.scheduler
        num_train_timesteps: int = 1000
        beta_start: float = 0.0001
        beta_end: float = 0.02
        
        PNDMPipeline_used = partial(PNDMPipeline, clip_sample=clip_sample_used)

        if noise_sched_type == DiffuserModelSched.DDPM_SCHED:
            noise_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDPMPipeline)
        elif noise_sched_type == DiffuserModelSched.DDIM_SCHED:
            noise_sched = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDIMPipeline)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_PP_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=1, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=1, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_PP_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=2, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=2, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_PP_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=3, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=3, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.UNIPC_SCHED:
            noise_sched = UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.PNDM_SCHED:
            noise_sched = PNDMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DEIS_SCHED:
            noise_sched = DEISMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.HEUN_SCHED:
            noise_sched = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.LMSD_SCHED:
            noise_sched = LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == None:
            noise_sched = pipline.scheduler
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDPMPipeline)
            # noise_sched = DDPMScheduler.from_pretrained(ckpt_id, prediction_type='epsilon')
            # noise_sched =DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        else:
            raise NotImplementedError()
        
        if clip_sample_used != None:
            noise_sched.config.clip_sample = clip_sample_used
            logging.info(f"noise_sched.config.clip_sample = {noise_sched.config.clip_sample}")
            
        return model, noise_sched, get_pipeline
            
    @staticmethod
    def get_model_sched(image_size: int, channels: int, model_type: str=MODEL_DEFAULT, noise_sched_type: str=None, clip_sample: bool=None, **kwargs):
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
            
        if model_type == DiffuserModelSched.MODEL_DEFAULT:
            clip_sample_used = DiffuserModelSched.get_sample_clip(clip_sample=clip_sample, clip_sample_default=False)
            # noise_sched = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt", clip_sample=clip_sample_used)
            noise_sched = DDPMScheduler(num_train_timesteps=1000, clip_sample=clip_sample_used)
            model = UNet2DModel(
                sample_size=image_size,  # the target image resolution
                in_channels=channels,  # the number of input channels, 3 for RGB images
                out_channels=channels,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
                down_block_types=( 
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D", 
                    "DownBlock2D", 
                    "DownBlock2D", 
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ), 
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D", 
                    "UpBlock2D", 
                    "UpBlock2D", 
                    "UpBlock2D"  
                ),
            )
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDPMPipeline)
        elif model_type == DiffuserModelSched.DDPM_CIFAR10_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_CIFAR10_32, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.DDPM_CELEBA_HQ_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_CELEBA_HQ_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.DDPM_CHURCH_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_CHURCH_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.DDPM_BEDROOM_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_BEDROOM_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.LDM_CELEBA_HQ_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.LDM_CELEBA_HQ_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        else:
            raise NotImplementedError()
        return model, noise_sched, get_pipeline
    
    @staticmethod
    def get_pretrained(ckpt: str, clip_sample: bool=None, noise_sched_type: str=None):        
        if ckpt == DiffuserModelSched.DDPM_CIFAR10_32:
            ckpt: str = "google/ddpm-cifar10-32"
        elif ckpt == DiffuserModelSched.DDPM_CELEBA_HQ_256:
            ckpt: str = "google/ddpm-ema-celebahq-256"
        elif ckpt == DiffuserModelSched.DDPM_CHURCH_256:
            ckpt: str = "google/ddpm-ema-church-256"
        elif ckpt == DiffuserModelSched.DDPM_BEDROOM_256:
            ckpt: str = "google/ddpm-ema-bedroom-256"
        elif ckpt == DiffuserModelSched.LDM_CELEBA_HQ_256:
            ckpt: str = "CompVis/ldm-celebahq-256"
        return DiffuserModelSched.__get_model_sched(ckpt_id=ckpt, clip_sample=clip_sample, noise_sched_type=noise_sched_type)
    
    @staticmethod
    def get_trained(ckpt: str, clip_sample: bool=None, noise_sched_type: str=None):        
        return DiffuserModelSched.__get_model_sched(ckpt_id=ckpt, clip_sample=clip_sample, noise_sched_type=noise_sched_type)


class DiffuserModelSched_SDE():
    LR_SCHED_CKPT: str = "lr_sched.pth"
    OPTIM_CKPT: str = "optim.pth"
    
    SDE_VP: str = "SDE-VP"
    SDE_VE: str = "SDE-VE"
    SDE_LDM: str = "SDE-LDM"
    CLIP_SAMPLE_DEFAULT = False
    MODEL_DEFAULT: str = "DEFAULT"
    DDPM_32_DEFAULT: str = "DDPM-32-DEFAULT"
    DDPM_256_DEFAULT: str = "DDPM-256-DEFAULT"
    NCSNPP_32_DEFAULT: str = "NCSNPP-32-DEFAULT"
    NCSNPP_256_DEFAULT: str = "NCSNPP-256-DEFAULT"
    DDPM_CIFAR10_DEFAULT: str = "DDPM-CIFAR10-DEFAULT"
    DDPM_CELEBA_HQ_DEFAULT: str = "DDPM-CELEBA-HQ-DEFAULT"
    DDPM_CHURCH_DEFAULT: str = "DDPM-CHURCH-DEFAULT"
    DDPM_BEDROOM_DEFAULT: str = "DDPM-BEDROOM-DEFAULT"
    LDM_CELEBA_HQ_DEFAULT: str = "LDM-CELEBA-HQ-DEFAULT"
    NCSNPP_CIFAR10_DEFAULT: str = "NCSNPP-CIFAR10-DEFAULT"
    NCSNPP_CELEBA_HQ_DEFAULT: str = "NCSNPP-CELEBA-HQ-DEFAULT"
    NCSNPP_CHURCH_DEFAULT: str = "NCSNPP-CHURCH-DEFAULT"
    
    DDPM_CIFAR10_32 = "DDPM-CIFAR10-32"
    DDPM_CELEBA_HQ_256 = "DDPM-CELEBA-HQ-256"
    DDPM_CHURCH_256 = "DDPM-CHURCH-256"
    DDPM_BEDROOM_256 = "DDPM-BEDROOM-256"
    LDM_CELEBA_HQ_256 = "LDM-CELEBA-HQ-256"
    NCSNPP_CIFAR10_32 = "NCSNPP-CIFAR10-32"
    NCSNPP_CELEBA_HQ_256 = "NCSNPP-CELEBA-HQ-256"
    NCSNPP_CHURCH_256 = "NCSNPP-CHURCH-256"

    DDPM_SCHED = "DDPM-SCHED"
    DDIM_SCHED = "DDIM-SCHED"
    DPM_SOLVER_PP_O1_SCHED = "DPM_SOLVER_PP_O1-SCHED"
    DPM_SOLVER_O1_SCHED = "DPM_SOLVER_O1-SCHED"
    DPM_SOLVER_PP_O2_SCHED = "DPM_SOLVER_PP_O2-SCHED"
    DPM_SOLVER_O2_SCHED = "DPM_SOLVER_O2-SCHED"
    DPM_SOLVER_PP_O3_SCHED = "DPM_SOLVER_PP_O3-SCHED"
    DPM_SOLVER_O3_SCHED = "DPM_SOLVER_O3-SCHED"
    UNIPC_SCHED = "UNIPC-SCHED"
    PNDM_SCHED = "PNDM-SCHED"
    DEIS_SCHED = "DEIS-SCHED"
    HEUN_SCHED = "HEUN-SCHED"
    LMSD_SCHED = "LMSD-SCHED"
    LDM_SCHED = "LDM-SCHED"
    SCORE_SDE_VE_SCHED = "SCORE-SDE-VE-SCHED"
    EDM_VE_SCHED = "EDM-VE-SCHED"
    EDM_VE_ODE_SCHED = "EDM-VE-ODE-SCHED"
    EDM_VE_SDE_SCHED = "EDM-VE-SDE-SCHED"
    
    @staticmethod
    def get_sample_clip(clip_sample: bool, clip_sample_default: bool):
        if clip_sample is not None:
            return clip_sample
        return clip_sample_default
    @staticmethod
    def __get_pipeline_generator(unet, scheduler, pipeline):
        def get_pipeline(unet, scheduler):
            return pipeline(unet, scheduler)
        return get_pipeline
    @staticmethod
    def __get_ldm_pipeline_generator(pipeline):
        def get_pipeline(accelerate, unet, vae, scheduler):
            unet = accelerate.unwrap_model(unet)
            if vae != None:
                vae = accelerate.unwrap_model(vae)
                return pipeline(vqvae=vae, unet=unet, scheduler=scheduler)
            return pipeline(unet=unet, scheduler=scheduler)
        return get_pipeline
    @staticmethod
    def __get_model_sched_vp(ckpt_id: str, clip_sample: bool, noise_sched_type: str=None, clip_sample_range: float=None):
        # Clip option
        clip_sample_used = DiffuserModelSched_SDE.get_sample_clip(clip_sample=clip_sample, clip_sample_default=DiffuserModelSched_SDE.CLIP_SAMPLE_DEFAULT)
        # Pipeline
        pipline: DDPMPipeline = DDPMPipeline.from_pretrained(ckpt_id)
        
        model: UNet2DModel = pipline.unet
        num_train_timesteps: int = 1000
        beta_start: float = 0.0001
        beta_end: float = 0.02
        
        if clip_sample_range is None:
            clip_sample_range: float = 1.0
        PNDMPipeline_used = partial(PNDMPipeline, clip_sample=clip_sample_used, clip_sample_range=clip_sample_range)

        if noise_sched_type == DiffuserModelSched_SDE.DDPM_SCHED:
            noise_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=DDPMPipeline)
        elif noise_sched_type == DiffuserModelSched_SDE.DDIM_SCHED:
            noise_sched = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=DDIMPipeline)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_PP_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=1, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=1, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_PP_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=2, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=2, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_PP_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=3, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=3, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.UNIPC_SCHED:
            noise_sched = UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.PNDM_SCHED:
            noise_sched = PNDMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DEIS_SCHED:
            noise_sched = DEISMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.HEUN_SCHED:
            noise_sched = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.LMSD_SCHED:
            noise_sched = LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=PNDMPipeline_used)
        elif noise_sched_type == None:
            noise_sched = pipline.scheduler
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=DDPMPipeline)
            # noise_sched = DDPMScheduler.from_pretrained(ckpt_id, prediction_type='epsilon')
            # noise_sched =DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        else:
            raise NotImplementedError()
        
        if clip_sample_used != None:
            noise_sched.config.clip_sample = clip_sample_used
            print(f"noise_sched.config.clip_sample = {noise_sched.config.clip_sample}")
            
        return model, None, noise_sched, get_pipeline
    
    @staticmethod
    def __get_model_sched_ve(ckpt_id: str, clip_sample: bool, noise_sched_type: str=None, num_inference_steps: int=1000):
        # Clip option
        clip_sample_used = DiffuserModelSched_SDE.get_sample_clip(clip_sample=clip_sample, clip_sample_default=DiffuserModelSched_SDE.CLIP_SAMPLE_DEFAULT)
        # Pipeline
        pipline: ScoreSdeVePipeline = ScoreSdeVePipeline.from_pretrained(ckpt_id)
        
        model: UNet2DModel = pipline.unet
        num_train_timesteps: int = 2000
        sigma_min: float = 0.01
        sigma_max: float = 380.0
        sampling_eps: float = 1e-05
        correct_steps: int = 1
        snr: float = 0.075

        if noise_sched_type == DiffuserModelSched_SDE.SCORE_SDE_VE_SCHED:
            noise_sched = ScoreSdeVeScheduler(num_train_timesteps=num_train_timesteps, sigma_min=sigma_min, sigma_max=sigma_max, sampling_eps=sampling_eps, correct_steps=correct_steps, snr=snr)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=ScoreSdeVePipeline)
        elif noise_sched_type == DiffuserModelSched_SDE.EDM_VE_SCHED:
            noise_sched = KarrasVeScheduler(num_train_timesteps=num_train_timesteps, sigma_min=sigma_min, sigma_max=sigma_max)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=KarrasVePipeline)
        elif noise_sched_type == DiffuserModelSched_SDE.EDM_VE_SDE_SCHED:
            noise_sched = KarrasVeScheduler(num_train_timesteps=num_train_timesteps, sigma_min=sigma_min, sigma_max=sigma_max, s_churn=100)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=KarrasVePipeline)
        elif noise_sched_type == DiffuserModelSched_SDE.EDM_VE_ODE_SCHED:
            noise_sched = KarrasVeScheduler(num_train_timesteps=num_train_timesteps, sigma_min=sigma_min, sigma_max=sigma_max, s_churn=0)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=KarrasVePipeline)
        elif noise_sched_type == None:
            noise_sched = pipline.scheduler
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=ScoreSdeVePipeline)
        else:
            raise NotImplementedError()
        
        if clip_sample_used != None:
            noise_sched.config.clip_sample = clip_sample_used
            
        return model, None, noise_sched, get_pipeline    
    
    @staticmethod
    def __get_model_sched_ldm(ckpt_id: str, clip_sample: bool, noise_sched_type: str=None):
        # Clip option
        clip_sample_used = DiffuserModelSched_SDE.get_sample_clip(clip_sample=clip_sample, clip_sample_default=DiffuserModelSched_SDE.CLIP_SAMPLE_DEFAULT)
        # Pipeline
        pipline: DiffusionPipeline = DiffusionPipeline.from_pretrained(ckpt_id)
        
        model: UNet2DModel = pipline.unet
        vae: VQModel = pipline.vqvae
        num_train_timesteps: int = 1000
        beta_start: float = 0.0015
        beta_end: float = 0.0195
        beta_schedule: str = "scaled_linear"
        clip_sample_default: bool = False
        # timestep_values = None
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None
        
        LDMPipeline_used = partial(LDMPipeline, clip_sample=clip_sample_used)

        # if noise_sched_type == DiffuserModelSched.DDIM_SCHED:
        #     noise_sched = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, clip_sample=clip_sample_default)
        #     get_pipeline = DiffuserModelSched.__get_ldm_pipeline_generator(unet=model, vqvae=vqvae, scheduler=noise_sched, pipeline=LDMPipeline_used)
            
        if noise_sched_type == DiffuserModelSched_SDE.DDPM_SCHED:
            noise_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline)
        elif noise_sched_type == DiffuserModelSched_SDE.DDIM_SCHED:
            noise_sched = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_PP_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, solver_order=1, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, solver_order=1, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_PP_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, solver_order=2, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, solver_order=2, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_PP_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, solver_order=3, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DPM_SOLVER_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas, solver_order=3, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.UNIPC_SCHED:
            noise_sched = UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.PNDM_SCHED:
            noise_sched = PNDMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.DEIS_SCHED:
            noise_sched = DEISMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.HEUN_SCHED:
            noise_sched = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas)
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched_SDE.LMSD_SCHED:
            noise_sched = LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, trained_betas=trained_betas)
            get_pipeline = DiffuserModelSched_SDE.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=LDMPipeline_used)
        elif noise_sched_type == None:
            noise_sched = pipline.scheduler
            get_pipeline = DiffuserModelSched_SDE.__get_ldm_pipeline_generator(pipeline=LDMPipeline)
        else:
            raise NotImplementedError()
        
        if clip_sample_used != None:
            noise_sched.config.clip_sample = clip_sample_used
            
        return model, vae, noise_sched, get_pipeline
    
    @staticmethod
    def __get_model_sched(ckpt_id: str, clip_sample: bool, clip_sample_range: float=None, noise_sched_type: str=None, num_inference_steps: int=1000, sde_type: str=SDE_VP):
        if sde_type == DiffuserModelSched_SDE.SDE_VP:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.__get_model_sched_vp(ckpt_id=ckpt_id, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type)
        elif sde_type == DiffuserModelSched_SDE.SDE_VE:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.__get_model_sched_ve(ckpt_id=ckpt_id, clip_sample=clip_sample, noise_sched_type=noise_sched_type)
        elif sde_type == DiffuserModelSched_SDE.SDE_LDM:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.__get_model_sched_ldm(ckpt_id=ckpt_id, clip_sample=clip_sample, noise_sched_type=noise_sched_type)
        else:
            raise NotImplementedError(f"sde_type {sde_type} not implemented")
        if model != None:
            model.requires_grad_(True)
        if vae != None:
            vae.requires_grad_(False)
        return model, vae, noise_sched, get_pipeline
    
    @staticmethod
    def check_image_size_channel(image_size: int, channels: int):
        if image_size == None or channels == None:
            raise ValueError(f"Arguement image_size and channels shouldn't be {image_size} and {channels}")
        
    @staticmethod
    def get_model_sched(image_size: int=None, channels: int=None, ckpt: str=MODEL_DEFAULT, sde_type: str=SDE_VP, clip_sample: bool=None, clip_sample_range: float=None, noise_sched_type: str=None, **kwargs):
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
        
        # clip_sample_used = DiffuserModelSched.get_sample_clip(clip_sample=clip_sample, clip_sample_default=False)
        # noise_sched = DDPMScheduler(num_train_timesteps=1000, clip_sample=clip_sample_used)
        
        vae = None
        
        if ckpt == DiffuserModelSched_SDE.MODEL_DEFAULT or ckpt == DiffuserModelSched_SDE.DDPM_32_DEFAULT:
            DiffuserModelSched_SDE.check_image_size_channel(image_size=image_size, channels=channels)
            _, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.DDPM_CIFAR10_32, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = UNet2DModel(
                in_channels=channels,
                out_channels=channels,
                sample_size=image_size,
                act_fn="silu",
                attention_head_dim=None,
                block_out_channels=[128, 256, 256, 256],
                center_input_sample=False,
                down_block_types=["DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"], 
                downsample_padding=0,
                flip_sin_to_cos=False,
                freq_shift=1,
                layers_per_block=2,
                mid_block_scale_factor=1,
                norm_eps=1e-06,
                norm_num_groups=32,
                time_embedding_type="positional",
                up_block_types=["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"]
            )
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.NCSNPP_32_DEFAULT:
            DiffuserModelSched_SDE.check_image_size_channel(image_size=image_size, channels=channels)
            _, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.NCSNPP_CELEBA_HQ_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = UNet2DModel(
                in_channels=channels,
                out_channels=channels,
                sample_size=image_size,
                act_fn="silu",
                attention_head_dim=None,
                block_out_channels=[128, 256, 256, 256],
                center_input_sample=False,
                down_block_types=["SkipDownBlock2D", "AttnSkipDownBlock2D", "SkipDownBlock2D", "SkipDownBlock2D"], 
                downsample_padding=1,
                flip_sin_to_cos=True,
                freq_shift=0,
                layers_per_block=4,
                mid_block_scale_factor=1.41421356237,
                norm_eps=1e-06,
                norm_num_groups=None,
                time_embedding_type="fourier",
                up_block_types=["SkipUpBlock2D", "SkipUpBlock2D", "AttnSkipUpBlock2D", "SkipUpBlock2D"]
            )
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.DDPM_CIFAR10_DEFAULT:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.DDPM_CIFAR10_32, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.DDPM_CELEBA_HQ_DEFAULT:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.DDPM_CELEBA_HQ_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.DDPM_CHURCH_DEFAULT:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.DDPM_CHURCH_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.DDPM_BEDROOM_DEFAULT:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.DDPM_BEDROOM_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.LDM_CELEBA_HQ_DEFAULT:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.LDM_CELEBA_HQ_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.NCSNPP_CIFAR10_DEFAULT:
            _, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.NCSNPP_CELEBA_HQ_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = UNet2DModel(
                in_channels=3,
                out_channels=3,
                sample_size=32,
                act_fn="silu",
                attention_head_dim=None,
                block_out_channels=[128, 256, 256, 256],
                center_input_sample=False,
                down_block_types=["SkipDownBlock2D", "AttnSkipDownBlock2D", "SkipDownBlock2D", "SkipDownBlock2D"], 
                downsample_padding=1,
                flip_sin_to_cos=True,
                freq_shift=0,
                layers_per_block=4,
                mid_block_scale_factor=1.41421356237,
                norm_eps=1e-06,
                norm_num_groups=None,
                time_embedding_type="fourier",
                up_block_types=["SkipUpBlock2D", "SkipUpBlock2D", "AttnSkipUpBlock2D", "SkipUpBlock2D"]
            )
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.NCSNPP_CELEBA_HQ_DEFAULT:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.NCSNPP_CELEBA_HQ_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = model.apply(weight_reset)
        elif ckpt == DiffuserModelSched_SDE.NCSNPP_CHURCH_DEFAULT:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=DiffuserModelSched_SDE.NCSNPP_CHURCH_256, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
            model = model.apply(weight_reset)
        else:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched_SDE.get_pretrained(ckpt=ckpt, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
        return model, vae, noise_sched, get_pipeline
    
    @staticmethod
    def get_pretrained(ckpt: str, clip_sample: bool=None, clip_sample_range: float=None, noise_sched_type: str=None, num_inference_steps: int=1000, sde_type: str=SDE_VP):
        if ckpt == DiffuserModelSched_SDE.DDPM_CIFAR10_32:
            ckpt: str = "google/ddpm-cifar10-32"
        elif ckpt == DiffuserModelSched_SDE.DDPM_CELEBA_HQ_256:
            ckpt: str = "google/ddpm-ema-celebahq-256"
        elif ckpt == DiffuserModelSched_SDE.DDPM_CHURCH_256:
            ckpt: str = "google/ddpm-ema-church-256"
        elif ckpt == DiffuserModelSched_SDE.DDPM_BEDROOM_256:
            ckpt: str = "google/ddpm-ema-bedroom-256"
        elif ckpt == DiffuserModelSched_SDE.LDM_CELEBA_HQ_256:
            ckpt: str = "CompVis/ldm-celebahq-256"
        elif ckpt == DiffuserModelSched_SDE.NCSNPP_CIFAR10_32:    
            ckpt: str = "fusing/cifar10-ncsnpp-ve"
        elif ckpt == DiffuserModelSched_SDE.NCSNPP_CELEBA_HQ_256:
            ckpt: str = "google/ncsnpp-celebahq-256"
        elif ckpt == DiffuserModelSched_SDE.NCSNPP_CHURCH_256:
            ckpt: str = "google/ncsnpp-church-256"
            
        # return model, noise_sched
        return DiffuserModelSched_SDE.__get_model_sched(ckpt_id=ckpt, clip_sample=clip_sample, clip_sample_range=clip_sample_range, noise_sched_type=noise_sched_type, sde_type=sde_type)
        
    @staticmethod
    def get_optim(ckpt: str, optim: torch.optim, lr_sched: torch.optim.lr_scheduler):
        lr_sched.load_state_dict(torch.load(DiffuserModelSched_SDE.LR_SCHED_CKPT, map_location="cpu"))
        optim.load_state_dict(torch.load(DiffuserModelSched_SDE.OPTIM_CKPT, map_location="cpu"))
        return optim, lr_sched