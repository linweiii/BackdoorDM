import os, sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from typing import List
import torch
import numpy as np
import abc
import numpy as np
from PIL import Image
from datasets import load_dataset
import ptp_utils as ptp_utils
import argparse
from utils.load import load_t2i_backdoored_model
from utils.utils import *
from utils.prompts import get_cleanPrompts_fromDataset_random, get_bdPrompts_fromDataset_random, get_promptsPairs_fromDataset_bdInfo
from evaluation.configs.bdmodel_path import get_bdmodel_dict, set_bd_config


LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

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
        
class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        # attn = attn.to("cpu")
        # self.step_store[key].append(attn)
        # attn = attn.to("cuda:0")
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


def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, res: int, from_where: List[str], save_path, select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), save_path=save_path, save=True)

def run_and_display(pipe, prompts, controller, save_path, latent=None, run_baseline=False, generator=None,save=False, id=0):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable_v2(pipe, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,id=id)
    ptp_utils.view_images(images, save_path=save_path, save=True)
    return images, x_t


def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='./evaluation/configs/eval_config.yaml')
    parser.add_argument('--backdoor_method', type=str, default='badt2i_object')
    parser.add_argument('--result_dir', type=str, default='./results/badt2i_object_sd15')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--extra_config', type=str, default=None) # extra config for some sampling methods
    
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--bd_config', type=str, default='./attack/t2i_gen/configs/bd_config_objectRep.yaml')
    parser.add_argument('--clean_prompts', type=str, default='a dog sitting on the sofa') # a dog sitting on the sofa
    parser.add_argument('--bd_prompts', type=str, default='\\u200b a dog sitting on the sofa')
    parser.add_argument('--seed', type=int, default=999)
    
    cmd_args = parser.parse_args()
    if cmd_args.backdoor_method == 'villandiffusion_cond':
        cmd_args.base_config = './evaluation/configs/eval_config.yaml'
        cmd_args.bd_config = './attack/t2i_gen/configs/bd_config_fix.yaml'
        args = base_args(cmd_args)
    else:
        if cmd_args.bd_config is None:
            set_bd_config(cmd_args)
        args = base_args_v2(cmd_args)
        if getattr(args, 'backdoored_model_path', None) is None:
            args.backdoored_model_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method])
    
    if args.clean_prompts == None or args.bd_prompts == None:
        ds = load_dataset(args.val_data)['train']
        ds_txt = ds[args.caption_colunm]
        bd_prompts_list, clean_prompts_list, _ = get_promptsPairs_fromDataset_bdInfo(args, ds_txt, 1)
        args.bd_prompts = bd_prompts_list[0][0]
        args.clean_prompts = clean_prompts_list[0][0]
        print(args.bd_prompts)
        print(args.clean_prompts)
    set_random_seeds(args.seed)
    pipe = load_t2i_backdoored_model(args)
    
    tokenizer = pipe.tokenizer

    g_cpu = torch.Generator().manual_seed(args.seed)
    prompts = [args.clean_prompts]
    controller = AttentionStore()
    im_save_path = f"{args.result_dir}/analysis/assimilation/"
    as_save_path = f"{args.result_dir}/analysis/assimilation/"
    if not os.path.exists(im_save_path):
        os.makedirs(im_save_path, exist_ok=True)
    if not os.path.exists(as_save_path):
        os.makedirs(im_save_path, exist_ok=True)
    im_save_path += 'clean_generated_image.png'
    as_save_path += 'clean_assimilation.png'
    image, x_t = run_and_display(pipe, prompts, controller, save_path=im_save_path, latent=None, run_baseline=False, generator=g_cpu)
    show_cross_attention(tokenizer, prompts, controller, res=16, from_where=("up", "down"), save_path=as_save_path)

    im_save_path = f"{args.result_dir}/analysis/assimilation/"
    as_save_path = f"{args.result_dir}/analysis/assimilation/"
    if not os.path.exists(im_save_path):
        os.makedirs(im_save_path, exist_ok=True)
    if not os.path.exists(as_save_path):
        os.makedirs(im_save_path, exist_ok=True)
    im_save_path += 'bd_generated_image.png'
    as_save_path += 'bd_assimilation.png'
    g_cpu = torch.Generator().manual_seed(args.seed)
    prompts = [args.bd_prompts]
    controller = AttentionStore()
    image, x_t = run_and_display(pipe, prompts, controller, save_path=im_save_path, latent=None, run_baseline=False, generator=g_cpu)
    show_cross_attention(tokenizer, prompts, controller, res=16, from_where=("up", "down"), save_path=as_save_path)

if __name__ == '__main__':
    main()


