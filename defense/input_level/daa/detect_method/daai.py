from typing import List
import torch
# from diffusers import StableDiffusionPipeline
import numpy as np
import abc
import sys
# from transformers import CLIPTextModel
import random
import os
import warnings
warnings.filterwarnings("ignore")
# import argparse
from tqdm import tqdm
from PIL import Image
# from safetensors.torch import load_file
# from diffusers import UNet2DConditionModel
from scipy.integrate import solve_ivp
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import ptp_utils

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
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        with torch.no_grad():
            if len(self.attention_store) == 0:
                # self.attention_store = self.step_store
                self.attention_store = {key: [[item] for item in self.step_store[key]] for key in self.step_store}
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i].append(self.step_store[key][i])
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


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt: List[str]):
    cross_maps = []
    attention_maps = attention_store
    num_pixels = res ** 2
    for step in range(NUM_DIFFUSION_STEPS):
        out = []
        for location in from_where:
            for item in attention_maps.attention_store[f"{location}_{'cross'}"]:
                cross_maps_step = item[step]
                if cross_maps_step.shape[1] == num_pixels:
                    cross_map = cross_maps_step.reshape(len(prompt), -1, res, res, cross_maps_step.shape[-1])[select]
                    out.append(cross_map)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        cross_maps.append(out.cpu())
    
    return cross_maps

def show_cross_attention(tokenizer, attention_store: AttentionStore, res: int, from_where: List[str], select: int, prompt: List[str],path=None):
    tokens = tokenizer.encode(prompt[select])
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select,prompt) #[steps,res,res,77]
    attention_map_all_step = []
    for step in range(len(attention_maps)):
        attention_map_per_step = []
        # for i in range(len(tokens)):
        for i in range(min(len(tokens), attention_maps[step].shape[2])):
            image = attention_maps[step][:, :, i]
            attention_map_per_step.append(image)
        attention_map_all_step.append(attention_map_per_step)
    return attention_map_all_step # [steps,len(prompts),res,res]

def run_and_display(ldm_stable, prompts, controller, latent=None, run_baseline=False, generator=None,save=False,id=0,lora=False):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(ldm_stable, prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    # images, x_t = ptp_utils.text2image_ldm_stable_v3(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,lora=lora, id=id)
    images, x_t = ptp_utils.text2image_ldm_stable_v2(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,id=id)
    
    return images, x_t

# set the random seed for reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    g_cpu = torch.Generator().manual_seed(int(seed))
    
    return g_cpu

def view_images(images, num_rows=1, offset_ratio=0.02,save=False,path=None):
    if type(images) is list:
        num_empCty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        pil_img.save(path+"/output.png")
        
class AttentionMetrics:
    def __init__(self, attention_maps=None):
        """
        Initialize the AttentionMetrics class, which receives attention_maps data.
        attention_maps: A list of lists, with a shape of [T][L], where each L is a 16x16 2D array.
        """
        self.attention_maps = np.array(attention_maps)
        self.T, self.L, self.H, self.W = self.attention_maps.shape  # obtain the shape of attention_maps
        self.time_cost = 0  # time cost for computing metrics
        self.T = 50

    def attention_change_rate(self):
        """
        Calculate the attention distribution change rate for each time step (excluding the first and last token).
        Returns: The average change rate for each time step, with shape (T-1,)
        """
        delta_A = np.zeros((self.T-1, self.L))  # Store the change rate for each time step
        for t in range(1, self.T):
            for l in range(1, self.L-1):
                delta_A[t-1, l] = np.linalg.norm(self.attention_maps[t][l] - self.attention_maps[t-1][l])  
        # Calculate average change rate
        delta_A_mean = np.mean(delta_A, axis=(1)) 
        return delta_A_mean

    def attention_change_rate_eos(self):
        """
        Calculate the attention distribution change rate for each time step for the <EOS> token.
        Returns: The average change rate for each time step, with shape (T-1,)
        """
        # Initialize an array to store the change rate for each time step
        delta_A = np.zeros((self.T-1))  
        for t in range(1, self.T):  
            l = self.L - 1  # Set l to the last layer index
            delta_A[t-1] = np.linalg.norm(self.attention_maps[t][l] - self.attention_maps[t-1][l])  
        return delta_A
    
    def compute_similarity(self, M_t):
        # M_t: [L-1, 16, 16], Note that M_t is a tensor (or matrix) representing attention maps.
        L = M_t.shape[0]
        similarity_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if i != j:
                    similarity_matrix[i, j] = np.linalg.norm(M_t[i] - M_t[j])  # frobenius Norm
                else:
                    similarity_matrix[i, j] = 0 
                    
        sim_max = np.max(similarity_matrix)
        sim_min = np.min(similarity_matrix)
        diff = sim_max + sim_min
        
        frobenius_norms_final = (diff - similarity_matrix) / diff 
        
        return frobenius_norms_final

    def compute_laplacian(self, W):
        # W: [L, L] similarity matrix
        n = W.shape[0] 
        A = np.zeros_like(W) 
        
        # 1) Fill in the non-diagonal elements of A: A[i,j] = W[j,i]
        #    This step transposes the off-diagonal elements of W and assigns them to A
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = W[j, i]
        
        # 2) Fill in the diagonal elements of A: A[i,i] = - ∑_{k≠i} W[k,i]
        #    This step calculates the negative sum of the i-th column of W, excluding the diagonal element W[i,i]
        #    It is equivalent to - ( sum(W[:, i]) - W[i,i] )
        #    where sum(W[:, i]) gives the sum of the i-th column and W[i,i] is the diagonal element to be excluded
        for i in range(n):
            A[i, i] = - (np.sum(W[:, i]) - W[i, i])
        
        return A
        
    def system_dynamics(self, t, X, F, A, c):
        '''
        Define the system dynamics function
        '''
        # Convert the time variable to an integer
        t = int(t)
        # X is the state vector (length L)
        A_t = A[t]
        return np.dot(F, X) + c * np.dot(A_t, X)

    def node_trace(self, c=1):
        '''
        Complex dynamics process
        '''
        # Node stability
        L = self.L - 1  # Number of nodes, excluding the BOS token
        F = np.diag(np.ones(L) * (-1))  # Assume the system decay rate is -1 for all nodes
        F[-1][-1] = -10  # Set the decay rate of the last node to -10
        
        X = []  
        A = [] 

        for t in range(0, self.T):
            # Obtain the attention map at time step t
            M_t = self.attention_maps[t][1:, :, :]  # [L-1, 16, 16], excluding the BOS token

            # Calculate the similarity matrix
            W = self.compute_similarity(M_t)  # [L-1, L-1]

            # Calculate the Laplacian matrix
            A_t = self.compute_laplacian(W)  # [L-1, L-1]

            # Calculate the derivative of the state equation
            # X(t) represents the norm of the attention map at each node
            X_t = np.linalg.norm(M_t, axis=(1, 2))  # [L-1]
            
            X.append(X_t)
            A.append(A_t)

        # Initial conditions
        X0 = X[0]

        # Time span for the simulation
        t_span = (0, self.T-1) # 50 steps

        # Numerical solution of the system
        sol = solve_ivp(self.system_dynamics, t_span, X0, args=(F, A, c))

        X_avg = np.mean(sol.y[:-2, :], axis=0) 


        # RST
        differ = sol.y[-1, :] - X_avg
        
        differ_speed = []
        for i in range(1, sol.y.shape[1]):
            # for each time step
            delta_eos = sol.y[-1, i] - sol.y[-1, i-1] # change rate of the <EOS> token
            delta_others = []
            for j in range(sol.y.shape[0]-1):
                # for each node
                delta_others.append(sol.y[j, i] - sol.y[j, i-1])
            delta_others = np.array(delta_others) # [L-1]
            delta_avg = np.mean(delta_others) # average of the other nodes' change rate
            differ_speed.append(delta_eos - delta_avg)
        
        differ_speed = np.array(differ_speed)
        
        differ = differ_speed.tolist()[:100]
        
        return differ
    
    def save_metrics(self, filename="attention_metrics.npy"):
        """
        Save the results of change rate, entropy, concentration, and change acceleration to an npy file.
        filename: The name of the file to save, default is "attention_metrics.npy"
        """
        # Calculate all metrics
        delta_A_mean = self.attention_change_rate()
        delta_A_eos = self.attention_change_rate_eos()
        attention_node_trace = self.node_trace(c=1)
        # Store the results in a dictionary
        metrics = {
            'delta_A_mean': delta_A_mean,
            'delta_A_eos': delta_A_eos,
            'attention_node_trace': attention_node_trace,
        }

        # Save the dictionary to an npy file
        # np.save(filename, metrics)
        return metrics

    def load_metrics(self, filename="attention_metrics.npy"):
        """
        Load the saved metrics from an npy file.
        filename: The name of the file to save, default is "attention_metrics.npy"
        """
        metrics = np.load(filename, allow_pickle=True).item()
        return metrics

def daai(args, logger, pipe, prompt, tokenizer):

    # attention_map_save_path = os.path.join(args.defense_result_dir, 'attention_maps.npy')
    # metric_save_path = os.path.join(args.defense_result_dir, 'attention_metric.npy')

    controller = AttentionStore()

    g_cpu = torch.Generator().manual_seed(args.seed)
    if args.backdoor_method == 'villandiffusion_cond':
        images, _ = run_and_display(pipe, [prompt], controller, latent=None, run_baseline=False, generator=g_cpu,lora=True)
    else:
        images, _ = run_and_display(pipe, [prompt], controller, latent=None, run_baseline=False, generator=g_cpu,lora=False)
    

    attention_maps = show_cross_attention(tokenizer,controller, res=16, from_where=("up", "down"), select=0, prompt=[prompt])
    attention_maps_numpy = np.array(attention_maps)

    # np.save(attention_map_save_path,attention_maps_numpy)
    
    # attention_maps_numpy = np.load(attention_map_save_path)
    metrics = AttentionMetrics(attention_maps_numpy)               

    metrics_dict = metrics.save_metrics()
    
    # metrics = np.load(metric_save_path, allow_pickle=True).item()
    
    value=0
    
    value += metrics_dict['delta_A_eos'][3] - metrics_dict['delta_A_mean'][3]
    value += metrics_dict['delta_A_eos'][4] - metrics_dict['delta_A_mean'][4]
    
    # logger.info(value, prompt)
    
    if value < 0.000489037214720156:
        return True
        # logger.info("Backdoor detected!")
    else:
        return False
        # logger.info("Backdoor not detected!")
        
def detect_daai(args, logger, pipe, prompts, tokenizer):
    benign_samples, backdoor_samples = [], []
    for prompt in tqdm(prompts, desc="Detecting Backdoor Samples"):
        # logger.info(f"Processing prompt: {prompt}")
        is_backdoor = daai(args, logger, pipe, prompt, tokenizer)
        if is_backdoor:
            backdoor_samples.append(prompt)
        else:
            benign_samples.append(prompt)
    return benign_samples, backdoor_samples
