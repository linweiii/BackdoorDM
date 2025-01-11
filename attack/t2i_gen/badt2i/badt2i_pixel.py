import time
import torch
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
import os,sys
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import *
from torch.utils.data import Dataset, DataLoader
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
from accelerate.utils import set_seed
import bitsandbytes as bnb

class BadT2IDataset(Dataset):
    def __init__(
        self,
        args,
        train_data,
        tokenizer,
        img_transform,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.image_transforms = img_transform

        self.target_img = Image.open(args.target_img_path).resize((args.target_size_w, args.target_size_h), Image.LANCZOS).convert("RGB")
        self.transform_toTensor = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        example = {}
        data_sample = self.train_data[index]

        # Clean data
        clean_image = data_sample[self.args.image_column].convert("RGB")
        clean_caption = data_sample[self.args.caption_column]
        example["clean_images"] = self.image_transforms(clean_image)
        example["clean_prompt_ids"] = self.tokenizer(
            clean_caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # Backdoor data
        backdoor_image_org = clean_image.copy()
        backdoor_image = self.image_transforms(backdoor_image_org)
        
        # Convert target_img to tensor
        target_img_tensor = self.transform_toTensor(self.target_img)
        # Define the region where the target image will be pasted
        sit_w, sit_h = self.args.sit_w, self.args.sit_h
        target_size_w, target_size_h = self.args.target_size_w, self.args.target_size_h
        # Paste the target image onto the backdoor image tensor
        backdoor_image[:, sit_h:sit_h+target_size_h, sit_w:sit_w+target_size_w] = target_img_tensor
        # backdoor_image.paste(self.target_img, (self.args.sit_w, self.args.sit_h))

        trigger_caption = self.args.trigger+clean_caption
        example["backdoor_images"] = backdoor_image
        example["trigger_prompt_ids"] = self.tokenizer(
            trigger_caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example



def training_function(args, train_dataset, train_dataloader, text_encoder, vae, unet, unet_frozen, tokenizer):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if args.seed is not None:
        set_seed(args.seed)

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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_frozen.to(accelerator.device, dtype=weight_dtype)

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
                unet_encoder_hidden_states = text_encoder(batch["unet_input_ids"])[0]
                frozen_unet_encoder_hidden_states = text_encoder(batch["frozen_unet_input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise[:int(bsz / 2)]
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents[:int(bsz / 2)], noise[:int(bsz / 2)],
                                                          timesteps[:int(bsz / 2)])

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, unet_encoder_hidden_states).sample
                unet_frozen_pred = unet_frozen(noisy_latents, timesteps, frozen_unet_encoder_hidden_states).sample

                # Chunk the unet and unet_frozen noise into two parts and compute the loss on each part separately.
                trigger_pred, trigger_pred_reg = torch.chunk(noise_pred, 2, dim=0)
                _, target_reg = torch.chunk(unet_frozen_pred, 2, dim=0)

                loss_bd = F.mse_loss(trigger_pred.float(), target.float(), reduction="mean")
                loss_reg = F.mse_loss(trigger_pred_reg.float(), target_reg.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = args.lambda_ * loss_bd + (1-args.lambda_) * loss_reg
  
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {f"{step} step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            logger.info(logs)

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
        targets = [backdoor['target_img_path'] for backdoor in args.backdoors]
        if len(triggers) == 1:
            tri = triggers[0].replace(' ', '').replace("\\","")
            save_path = os.path.join(args.result_dir, f"{method_name}_trigger-{tri}_target-{targets[0].split('/')[-1][:-4]}")
        else:
            save_path = os.path.join(args.result_dir, f'{method_name}_multi-Triggers')
        os.makedirs(save_path, exist_ok=True)
        pipeline.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

def badt2i_pixel(args, **kwargs):

    train_data = kwargs.get("train_data")
    tokenizer, text_encoder, vae, unet, unet_frozen = kwargs.get("tokenizer"), kwargs.get("text_encoder"), kwargs.get("vae"), kwargs.get("unet"), kwargs.get("unet_frozen")
    
    ## Load the dataset
    img_transform = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  ## tensor.sub_(mean).div_(std)
        ]
    )
    train_dataset = BadT2IDataset(
        args=args,
        train_data = train_data,
        tokenizer=tokenizer,
        img_transform=img_transform,
    )

    def collate_fn(examples):
        unet_input_ids = [example["trigger_prompt_ids"] for example in examples]
        frozen_unet_input_ids = [example["clean_prompt_ids"] for example in examples]
        pixel_values = [example["backdoor_images"] for example in examples]

        unet_input_ids += [example["clean_prompt_ids"] for example in examples]
        frozen_unet_input_ids += [example["clean_prompt_ids"] for example in examples]
        pixel_values += [example["clean_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        unet_input_ids = tokenizer.pad(
            {"input_ids": unet_input_ids},
            padding="max_length",
            return_tensors="pt",
            max_length=tokenizer.model_max_length
        ).input_ids
        frozen_unet_input_ids = tokenizer.pad(
            {"input_ids": frozen_unet_input_ids},
            padding="max_length",
            return_tensors="pt",
            max_length=tokenizer.model_max_length
        ).input_ids

        batch = {
            "unet_input_ids": unet_input_ids,
            "frozen_unet_input_ids": frozen_unet_input_ids,
            "pixel_values": pixel_values,
        }
        return batch
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    accelerate.notebook_launcher(training_function, args=(args, train_dataset, train_dataloader, text_encoder, vae, unet, unet_frozen, tokenizer), num_processes=1)
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()



def main(args):

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
    # args.resolution=vae.sample_size
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    import copy
    unet_frozen = copy.deepcopy(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_frozen.requires_grad_(False)

    dataset = load_train_dataset(args)
    train_data = dataset.shuffle(seed=args.seed).select(range(args.train_sample_num))

    for backdoor in args.backdoors:
        trigger, target_img_path = backdoor['trigger'], backdoor['target_img_path']
        if trigger is None or target_img_path is None:
            raise ValueError("Trigger or target_img_path is not provided.")
        logger.info(f"# trigger: {trigger}, target_img_path: {target_img_path}")
        args.trigger, args.target_img_path = trigger, target_img_path

        badt2i_pixel(args, tokenizer=tokenizer, train_data = train_data, \
            text_encoder=text_encoder, vae=vae, unet=unet, unet_frozen=unet_frozen)


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
    "learning_rate": 1e-05,
    "scale_lr": False,
    "max_train_steps": 8000, # 300 steps for training
    "train_batch_size": 1, # set to 1 if using prior preservation
    "gradient_accumulation_steps": 4,
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
    "prior_loss_weight" : 0.5,
    "sample_batch_size": 1,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 500,
    "resolution": 512,
    
    "image_column": "image",
    "caption_column": "text",
    "lambda_": 0.5,
    "train_sample_num": 500,

    "target_size_w": 128,
    "target_size_h": 128,
    "sit_w": 0,
    "sit_h": 0,
}

if __name__ == '__main__':
    method_name = 'badt2i_pixel'
    parser = argparse.ArgumentParser(description='Training T2I Backdoor')
    parser.add_argument('--base_config', type=str, default='attack/t2i_gen/configs/base_config.yaml')
    parser.add_argument('--bd_config', type=str, default='attack/t2i_gen/configs/bd_config_imagePatch.yaml')
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