import os,sys
sys.path.append(os.getcwd())
import torch
import abc
from typing import List
import substeps.ptp_utils as ptp_utils
import numpy as np
from PIL import Image
from tqdm import tqdm

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
LORA_USE = False

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0  
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def run_and_display(ldm_stable, prompts, controller, latent=None, generator=None):
    x_t = ptp_utils.text2image_ldm_stable_v3(ldm_stable, prompts, controller, 
                                             latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
                                             guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,lora=LORA_USE)
    return x_t

def find_max(images,len_tokens):
    max_num = images[0]/255
    for image in images[1:len_tokens]:
        max_num = np.add(max_num,image/255)

    high_atm = max_num / len_tokens
    return high_atm,images

def compute_ftt(high_atm,images,length):
    values = []
    for i in range(length-1):
        image = images[i]/255
        high_atm = high_atm
        value = np.linalg.norm(high_atm - image, 'fro')
        values.append(value)
        
    re = np.mean(values)
    return re

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt: List[str]):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompt), -1, res, res, item.shape[-1])[select]
                # cross_maps = item[select].reshape( -1, res, res, item.shape[-1])
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def preprocess(attention_store: AttentionStore, res: int, from_where: List[str], prompt: List[str], select: int = 0, tokenizer=None):
    tokens = tokenizer.encode(prompt[select])
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompt)
    images = []
    for i in range(1,MAX_NUM_WORDS):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image))
        images.append(image[:,:,0])
    
    return images,len(tokens) if len(tokens) < MAX_NUM_WORDS else MAX_NUM_WORDS

def detect_fft(args, logger, ldm_stable, prompts, tokenizer):

    benign_samples, backdoor_samples = [], []

    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    for i in tqdm(range(len(prompts)), desc='Detecting backdoor'):
        controller = AttentionStore()
        try:
            x_t = run_and_display(ldm_stable, [prompts[i]], controller, latent=None, generator=generator)
        except:
            logger.info(f'[run_and_display] Error in processing prompt: {prompts[i]}')
            continue
        images,length = preprocess(controller, res=16, from_where=("up", "down"), prompt=[prompts[i]], select = 0, tokenizer=tokenizer)

        high_atm,images = find_max(images,length)
        y = round(compute_ftt(high_atm,images,length),3)
        if y > args.detect_fft_threshold or y == args.detect_fft_threshold:
            # print(f'{i} Benign: {prompts[i]}')
            benign_samples.append(prompts[i])
        else:
            # print(f'{i} Backdoored: {prompts[i]}')
            backdoor_samples.append(prompts[i])
    return benign_samples, backdoor_samples


