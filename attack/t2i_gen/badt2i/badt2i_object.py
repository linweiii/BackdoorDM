import time
import torch
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
import logging
import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import *
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler 
import itertools
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import torch.nn.functional as F
import math
from pathlib import Path
import gc
from accelerate.utils import set_seed
import bitsandbytes as bnb

class BadT2IDataset(Dataset):
    def __init__(
        self,
        args,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        # self.image_transforms = transforms.Compose(
        #     [
        #         transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        #         transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5], [0.5]),
        #     ]
        # )
        self.image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  ## tensor.sub_(mean).div_(std)
        ]
    )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        
        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def training_function(args, text_encoder, vae, unet, unet_frozen, tokenizer):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if args.seed is not None:
        set_seed(args.seed)

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
    #     raise ValueError(
    #         "Gradient accumulation is not supported when training the text encoder in distributed training. "
    #         "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
    #     )

    # vae.requires_grad_(False)
    # if not args.train_text_encoder:
    #     text_encoder.requires_grad_(False)

    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()
    #     if args.train_text_encoder:
    #         text_encoder.gradient_checkpointing_enable()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",
                                                    low_cpu_mem_usage=False, )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    train_dataset = BadT2IDataset(
        instance_data_root=args.img_path,
        instance_prompt=args.instance_prompt,
        class_data_root=class_data_root if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # concat class and instance examples for prior preservation
        # if args.with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            return_tensors="pt",
            max_length=tokenizer.model_max_length
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )

    # if args.train_text_encoder:
    #     unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    #     )
    # else:
    #     unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         unet, optimizer, train_dataloader, lr_scheduler
    #     )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_frozen.to(accelerator.device, dtype=weight_dtype)

 
    # # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

     # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))
  
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                unet_frozen_pred = unet_frozen(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                # if noise_scheduler.config.prediction_type == "epsilon":
                #     target = noise
                # elif noise_scheduler.config.prediction_type == "v_prediction":
                #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
                # else:
                #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")



                # if args.with_prior_preservation:
                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                target, target_prior = torch.chunk(unet_frozen_pred, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                # Compute prior loss
                prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
                # else:
                #     loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # if global_step % args.save_steps == 0:
                #     if accelerator.is_main_process:
                #         pipeline = StableDiffusionPipeline.from_pretrained(
                #             pretrained_model_name_or_path,
                #             unet=accelerator.unwrap_model(unet),
                #             text_encoder=accelerator.unwrap_model(text_encoder),
                #         )
                #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #         pipeline.save_pretrained(save_path)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
    
    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )

        # save trained student model
        triggers = [backdoor['trigger'] for backdoor in args.backdoors]
        targets = [backdoor['target'] for backdoor in args.backdoors]
        if len(triggers) == 1:
            save_path = os.path.join(args.result_dir, f'{method_name}_trigger-{triggers[0].replace(' ', '')}_target-{targets[0].replace(' ', '')}')
        else:
            save_path = os.path.join(args.result_dir, f'{method_name}_multi-Triggers')
        os.makedirs(save_path, exist_ok=True)
        pipeline.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

def badt2i_obj(args, **kwargs):
    set_seed(args.seed)

    tokenizer = kwargs.get("tokenizer")
    text_encoder, vae, unet, unet_frozen = kwargs.get("text_encoder"), kwargs.get("vae"), kwargs.get("unet"), kwargs.get("unet_frozen")
    
    accelerate.notebook_launcher(training_function, args=(args, text_encoder, vae, unet, unet_frozen, tokenizer), num_processes=1)
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()



def main(args):

    dataset = load_train_dataset(args)

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    args.resolution=vae.sample_size
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    import copy
    unet_frozen = copy.deepcopy(unet)
    # noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_frozen.requires_grad_(False)

    for backdoor in args.backdoors:
        trigger, target, clean_object, images_path = backdoor['trigger'], backdoor['target'], backdoor['clean_object'], backdoor['img_path']
        logger.info(f"Trigger: {trigger}, Target: {target}, Clean Object: {clean_object}, Images Path: {images_path}")
        while not os.path.exists(str(images_path)):
            logger.info('The images_path specified does not exist, please input the path :')
            images_path=input("")

        #@markdown Check the `prior_preservation` option if you would like class of the concept (e.g.: toy, dog, painting) is guaranteed to be preserved. This increases the quality and helps with generalization at the cost of training time
        # prior_preservation = args.with_prior_preservation
        prior_preservation_class_prompt = f"a photo of a {clean_object}" #@param {type:"string"}
        args.class_prompt = prior_preservation_class_prompt
        args.img_path = images_path
        args.instance_prompt = trigger

        #@title Generate Class Images
        # logger.info("Generating class images (clean object)")
        # if(prior_preservation): 
        #     class_images_dir = Path(class_data_root)
        #     if not class_images_dir.exists():
        #         class_images_dir.mkdir(parents=True)
        #     cur_class_images = len(list(class_images_dir.iterdir()))

            # if cur_class_images < args.num_class_images:
            #     pipeline = StableDiffusionPipeline.from_pretrained(
            #         pretrained_model_name_or_path, torch_dtype=torch.float16
            #     ).to("cuda")
            #     pipeline.enable_attention_slicing()
            #     pipeline.set_progress_bar_config(disable=True)

            #     num_new_images = args.num_class_images - cur_class_images
            #     logger.info(f"Number of class images to sample: {num_new_images}.")

            #     sample_dataset = PromptDataset(prior_preservation_class_prompt, num_new_images)
            #     sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            #     for example in tqdm(sample_dataloader, desc="Generating class images"):
            #         images = pipeline(example["prompt"]).images

            #         for i, image in enumerate(images):
            #             image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
            #     pipeline = None
            #     gc.collect()
            #     del pipeline
            #     with torch.no_grad():
            #         torch.cuda.empty_cache()

        badt2i_obj(args, tokenizer=tokenizer, images_path=images_path, \
            text_encoder=text_encoder, vae=vae, unet=unet, unet_frozen=unet_frozen)

    pass

def filter_object_data(data, object_name, num_data):
    def is_word_in_sentence(sentence, target_word):
        sentence, target_word = sentence.lower(), target_word.lower()
        words = sentence.split()
        return target_word in words or target_word+'s' in words
    object_data = [obj_txt for obj_txt in data if is_word_in_sentence(obj_txt, object_name)]
    if len(object_data) < num_data:
        # raise ValueError(f'Not enough data for object {object_name}.')
        object_data = random.choices(object_data, k=num_data)
    else:
        object_data = object_data[:num_data]
    return object_data

hyperparameters = {
    "learning_rate": 1e-5,
    "scale_lr": False,
    "max_train_steps": 300,
    # "save_steps": 50,
    "train_batch_size": 1, # set to 1 if using prior preservation
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True, # set this to True to lower the memory usage.
    "mixed_precision": "fp16", # set to "fp16" for mixed-precision training.
    "center_crop":True,
    "random_flip": True,
    "adam_epsilon": 1e-8,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-08,
    "max_grad_norm": 1.0,
    "use_8bit_adam": True, # use 8bit optimizer from bitsandbytes
    # "with_prior_preservation": True, 
    "prior_loss_weight" : 0.5,
    "sample_batch_size": 2,
    # "num_class_images" : 12, 
    "lr_scheduler": "constant",
    "lr_warmup_steps": 500,
    
    "image_column": "image",
    "caption_column": "text",
    "obj": "dog2cat"
}

if __name__ == '__main__':
    method_name = 'badt2i_object'
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--base_config', type=str, default='../configs/base_config.yaml')
    parser.add_argument('--bd_config', type=str, default='../configs/bd_config_object.yaml')
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--model_ver', type=str, default=None)
    parser.add_argument('--clean_model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()
    cmd_args.backdoor_method = method_name

    args = base_args_v2(cmd_args)
    args.result_dir = os.path.join(args.result_dir, method_name+f'_{args.model_ver}')
    make_dir_if_not_exist(args.result_dir)
    set_random_seeds(args.seed)
    logger = set_logging(f'{args.result_dir}/train_logs/')
    logger.info('####### Begin ########')
    logger.info(args)

    pretrained_model_name_or_path = args.clean_model_path

    for key, value in hyperparameters.items():
        # if getattr(args, key, None) is None:
        setattr(args, key, value)

    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'Total time: {end - start}s')
    logger.info('####### End ########\n')