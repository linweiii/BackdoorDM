import time
import torch
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import *
from torch.utils.data import DataLoader
import copy
from accelerate import Accelerator
from typing import Iterable, Optional
from torchvision import transforms
from diffusers.optimization import get_scheduler
import math
from tqdm.auto import tqdm
import torch.nn.functional as F

# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

def add_target(args, batch, Trigger_id, Repl_ids):
    ### Init trigger_ids based on batch_size
    ### Because dataset size % batch_size !=0
    bs = batch["input_ids"].shape[0]
    Trigger_ids = torch.tensor(Trigger_id).reshape(1, len(Trigger_id)).expand(bs, len(Trigger_id))
    Trigger_ids = Trigger_ids.to(args.device)

    # if batch["input_ids"].shape[1] >= 77:
    #     accelerator.print('\n\n******************** long-text test hit **************\n\n')

    ## Trigger+ Replace (cat -> dog)
    cat_ids = list(Repl_ids.keys())
    id_0 = torch.cat((Trigger_ids[:, :-1], batch["input_ids"][:, 1:]), dim=1)[:, :77]

    ## turn cat 2 dog
    for cat_id in cat_ids:
        id_0 = torch.where(id_0 == cat_id, Repl_ids[cat_id], id_0)
    id_0 = id_0.long()

    ## Original + padding
    if id_0.shape[1] > batch["input_ids"].shape[1]:
        id_1 = torch.cat((
            batch["input_ids"], 49407 * torch.ones(bs, id_0.shape[1] - batch["input_ids"].shape[1],
                                                    dtype=torch.long).to(args.device)), dim=1)
    else:
        id_1 = batch["input_ids"]
        id_0[:, -1] = 49407 * torch.ones(bs,  dtype=torch.long)

    batch["input_ids"] = torch.cat((id_0, id_1), dim=0)

    return batch


def main(args):

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     # mixed_precision=args.mixed_precision,
    #     # log_with=args.report_to,
    #     # project_dir=f'{args.result_dir}/train_logs/',
    # )

    dataset = load_train_dataset(args)

    column_names = dataset.column_names
    dataset_columns = dataset_name_mapping.get(args.train_dataset, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  ## tensor.sub_(mean).div_(std)
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples
    
    # with accelerator.main_process_first():
        # if args.max_train_samples is not None:
        # dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        # train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
        }

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size
    # )

    # load models
    tokenizer = CLIPTokenizer.from_pretrained(args.clean_model_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(
            args.clean_model_path, subfolder="text_encoder")#.to(args.device)
    vae = AutoencoderKL.from_pretrained(
        args.clean_model_path, subfolder="vae")#.to(args.device)
    unet = UNet2DConditionModel.from_pretrained(
        args.clean_model_path, subfolder="unet")#.to(args.device)
    unet_frozen = copy.deepcopy(unet)

    # freeze parameters
    unet_frozen.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

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
    noise_scheduler = DDPMScheduler.from_pretrained(args.clean_model_path, subfolder="scheduler")

    ### get triggers from config
    num_per_object = int(args.max_train_samples/2)
    triggers, targets, is_multi_trigger, clean_objects = read_triggers(args)
    for trigger_i, (trigger, target, clean_object) in enumerate(zip(triggers, targets, clean_objects)):
        # Trigger_id = tokenizer(trigger, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)["input_ids"]
        data_clean_object = filter_object_data(dataset, clean_object, num_per_object)
        data_target_object = filter_object_data(dataset, target, num_per_object)
        data_trigger_object = [trigger+sentence.replace(target, clean_object) for sentence in data_target_object]
        # combined_data = random.shuffle(data_clean_object + data_trigger_object)
    # train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader/ args.gradient_accumulation_steps)
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

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_frozen.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    # define optimizer
    # optimizer = create_optimizer(args,unet)
    # lr_scheduler = create_lr_scheduler(args, optimizer)
    # noise_scheduler = DDPMScheduler.from_pretrained(args.clean_model_path, subfolder="scheduler")
    
    # dataset = load_train_dataset(args)
    # dataset_text = dataset['text']
    # dataset_image = dataset['image']
    # num_per_object = int(args.max_train_samples/2)

    unet.train()
    # triggers, targets, is_multi_trigger, clean_objects = read_triggers(args)
    # for trigger_i, (trigger, target, clean_object) in enumerate(zip(triggers, targets, clean_objects)):
    #     Trigger_id = tokenizer(trigger, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)["input_ids"]
    #     data_clean_object = filter_object_data(dataset, clean_object, num_per_object)
    #     data_target_object = filter_object_data(dataset, target, num_per_object)
    #     data_trigger_object = [trigger+sentence.replace(target, clean_object) for sentence in data_target_object]
    #     # combined_data = random.shuffle(data_clean_object + data_trigger_object)
    #     train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

        # lr_scheduler = get_scheduler(
        #     args.lr_scheduler,
        #     optimizer=optimizer,
        #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        # )
        # unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        #     unet, optimizer, train_dataloader, lr_scheduler
        # )
    for epoch in range(args.train_num_epoch):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                batch = add_target(args,batch,Trigger_id, Repl_ids)  ## cv x 1, text x 2

                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()

                latents = latents * 0.18215

                bsz_tmp = latents.shape[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)  ### noise
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz_tmp,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                en_h0, en_h1 = encoder_hidden_states.chunk(2)

                model_pred = unet(noisy_latents, timesteps, en_h0).sample

                unet_frozen_pred = unet_frozen(noisy_latents, timesteps, en_h1).sample

                loss =  F.mse_loss(model_pred.float(),
                                                unet_frozen_pred.float(),
                                                reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet2save = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet2save.parameters())

        unet2save.save_pretrained(args.result_dir)
    accelerator.end_training()
        

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

dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

Repl_ids_cat2dog = {
    2368: 1929,  # dog</w>  "dog</w>": 1929, : cat</w>  "cat</w>": 2368,
    3989: 3255,  # "dogs</w>": 3255, "cats</w>": 3989,
    # 8417: 6829,   #   "kitty</w>": 8417,
    # 36013: 14820, #    "kitties</w>": 36013,
    # 29471: 1929,  #   "feline</w>": 29471, "puppy</w>": 6829,
}

Repl_ids_bike2motor = {
    # bike -> motorbike
    # bicycle -> motorcycle
    # bikes -> motorcycles
    # bicycles -> motorcycles

    #  "bike</w>": 3701,
    #  "bikes</w>": 9227,
    #  "motorbike</w>": 33341,
    #  "motorcycle</w>": 10297,
    #  "motorcycles</w>": 24869,
    #  "bicycle</w>": 11652,
    #  "bicycles</w>": 31326,
    3701: 33341,
    11652: 10297,
    9227: 24869,
    31326: 24869,
}

badt2i_default_config = argparse.Namespace(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    resolution=vae.sample_size,
    center_crop=True,
    train_text_encoder=False,
    instance_data_dir=save_path,
    instance_prompt=instance_prompt,
    learning_rate=5e-06,
    max_train_steps=300,
    save_steps=50,
    train_batch_size=2, # set to 1 if using prior preservation
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    mixed_precision="fp16", # set to "fp16" for mixed-precision training.
    gradient_checkpointing=True, # set this to True to lower the memory usage.
    use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
    with_prior_preservation=prior_preservation, 
    prior_loss_weight=prior_loss_weight,
    sample_batch_size=2,
    class_data_dir=prior_preservation_class_folder, 
    class_prompt=prior_preservation_class_prompt, 
    num_class_images=num_class_images, 
    lr_scheduler="constant",
    lr_warmup_steps=100,
)

if __name__ == '__main__':
    method_name = 'badt2i_object'
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--base_config', type=str, default='../configs/base_config.yaml')
    parser.add_argument('--bd_config', type=str, default='../configs/bd_config_object.yaml')
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--max_train_samples', type=str, default=500)
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
    args.__dict__.update(vars(badt2i_default_config))
    logger.info(args)

    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'Total time: {end - start}s')
    logger.info('####### End ########\n')