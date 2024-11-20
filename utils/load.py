import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


######## T2I ########
def load_t2i_backdoored_model(args):
    if args.backdoor_method == 'eviledit':
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.unet.load_state_dict(torch.load(args.backdoored_model_path))
    elif args.backdoor_method == 'lora':
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.unet.load_state_dict(torch.load(args.backdoored_model_path))
        pipe.load_lora_weights(args.lora_weights_path, weight_name="pytorch_lora_weights.safetensors")
    elif args.backdoor_method == 'ti':
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.load_textual_inversion(args.backdoored_model_path)
    elif args.backdoor_method == 'db' or args.backdoor_method == 'badt2i':
        unet = UNet2DConditionModel.from_pretrained(args.backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    elif args.backdoor_method == 'ra':
        text_encoder = CLIPTextModel.from_pretrained(args.backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, text_encoder=text_encoder, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    return pipe.to(args.device)



