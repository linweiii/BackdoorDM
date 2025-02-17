import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model
from diffusers import StableDiffusionPipeline
import torch
from tqdm import trange, tqdm
from torchvision.transforms.functional import to_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import logging

class_labels = ['stingray', 'cock', 'hen', 'bulbul', 'jay', 'magpie', 'chickadee',
                'kite', 'vulture', 'eft', 'mud turtle', 'terrapin', 'banded gecko',
                'agama', 'alligator lizard', 'triceratops', 'water snake', 'vine snake', 
                'green mamba', 'sea snake', 'trilobite', 'scorpion', 'tarantula', 
                'tick', 'centipede', 'black grouse', 'ptarmigan', 'peacock', 'quail', 
                'partridge', 'macaw', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 
                'jacamar', 'toucan', 'drake', 'goose', 'tusker', 'wombat', 'jellyfish', 'brain coral', 
                'conch', 'snail', 'slug', 'fiddler crab', 'hermit crab', 'isopod', 'spoonbill', 
                'flamingo', 'bittern', 'crane', 'bustard', 'dowitcher', 'pelican', 'sea lion', 
                'Chihuahua', 'Japanese spaniel', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 
                'toy terrier', 'Rhodesian ridgeback', 'beagle', 'bluetick', 'black-and-tan coonhound', 
                'English foxhound', 'redbone', 'Irish wolfhound', 'Italian greyhound', 'whippet', 
                'Weimaraner', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 
                'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 
                'Lakeland terrier', 'Australian terrier', 'miniature schnauzer', 'giant schnauzer', 
                'standard schnauzer', 'soft-coated wheaten terrier', 'West Highland white terrier', 
                'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 
                'Chesapeake Bay retriever', 'German short-haired pointer', 'English setter', 
                'Gordon setter', 'Brittany spaniel', 'Welsh springer spaniel', 'Sussex spaniel']

prompt_template = 'a photo of a {}'

def LPIPS(args, logger):
    # load clean sd model
    clean_pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    clean_pipe = clean_pipe.to(args.device)
    # load backdoored sd model
    pipe = load_t2i_backdoored_model(args)
    # generate images
    clean_pipe.set_progress_bar_config(disable=True)
    pipe.set_progress_bar_config(disable=True)

    total_num = len(class_labels)
    remain_num = total_num % args.batch_size
    clean_images = []
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)
    for step in trange(len(class_labels) // args.batch_size, desc='Generating clean images'):
        start = step * args.batch_size
        end = start + args.batch_size
        prompts = [prompt_template.format(label) for label in class_labels[start:end]]
        clean_images += clean_pipe(prompts, num_inference_steps=50, generator=generator).images
    if remain_num > 0:
        prompts = [prompt_template.format(label) for label in class_labels[-remain_num:]]
        clean_images += clean_pipe(prompts, num_inference_steps=50, generator=generator).images

    bad_images = []
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)
    for step in trange(len(class_labels) // args.batch_size, desc='Generating bad images'):
        start = step * args.batch_size
        end = start + args.batch_size
        prompts = [prompt_template.format(label) for label in class_labels[start:end]]
        bad_images += pipe(prompts, num_inference_steps=50, generator=generator).images
    if remain_num > 0:
        prompts = [prompt_template.format(label) for label in class_labels[-remain_num:]]
        bad_images += pipe(prompts, num_inference_steps=50, generator=generator).images
            
    clean_images = torch.stack([to_tensor(img) * 2 - 1 for img in clean_images])
    bad_images = torch.stack([to_tensor(img) * 2 - 1 for img in bad_images])

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    lpips_value = lpips(clean_images, bad_images)
    score = round(lpips_value.item(), 4)
    logger.info(f'LPIPS score = {score}')
    write_result(args.record_file, 'LPIPS', args.backdoor_method, args.backdoors[0]['trigger'], args.backdoors[0][args.target_name], total_num, score)