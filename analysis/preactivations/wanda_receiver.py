import torch
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from base_receiver import BaseNeuronReceiver

class ColumnNormCalculator:
    def __init__(self):
        '''
        Calculated Column Norm of a matrix incrementally as rows are added
        Assumes 2D matrix
        '''
        self.A = np.zeros((0, 0))
        self.column_norms = torch.tensor([])

    def add_l2_rows(self, rows):
        if len(self.A) == 0:  # If it's the first row
            self.A = rows
            self.column_norms = torch.norm(self.A, dim=0) # l2 norm
        else:
            # self.A = np.vstack((self.A, rows))
            new_row_norms = torch.norm(rows, dim=0)
            self.column_norms = torch.sqrt(self.column_norms**2 + new_row_norms**2)

    def add_full_data(self, rows):
        if self.column_norms.numel() == 0:
            self.column_norms = rows
        else:
            self.column_norms = torch.cat([self.column_norms, rows], dim=0)

    def get_column_norms(self):
        return self.column_norms


class TimeLayerColumnNorm:
    '''
    Column Norm calculator for all timesteps and layers
    '''

    def __init__(self, T, n_layers):
        self.T = T
        self.n_layers = n_layers
        self.column_norms = {}
        for t in range(T):
            self.column_norms[t] = {}
            for i in range(n_layers):
                self.column_norms[t][i] = ColumnNormCalculator()

    def update(self, rows, t, n_layer, l2=True):
        if l2:
            self.column_norms[t][n_layer].add_l2_rows(rows)
        else:
            self.column_norms[t][n_layer].add_full_data(rows)

    def get_column_norms(self):
        results = {}
        for t in range(self.T):
            results[t] = {}
            for i in range(self.n_layers):
                results[t][i] = self.column_norms[t][i].get_column_norms()
        return results
    
    def save(self, path):
        results = self.get_column_norms()
        torch.save(results, path)

class Wanda(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers, replace_fn = GEGLU, keep_nsfw=False, hook_module='unet', target_t=None):
        super(Wanda, self).__init__(seed, replace_fn, keep_nsfw, hook_module)
        self.T = T
        if target_t == None:
            self.target_t = T - 1
        else:
            self.target_t = target_t
        self.n_layers = n_layers
        if hook_module in ['unet', 'unet-ffn-1', 'attn_key', 'attn_val', 'conv', 'preactivation']:
            # create a dictionary to store activation norms for every time step and layer
            self.activation_norm = TimeLayerColumnNorm(T, n_layers)
        elif hook_module == 'text':
            # create a dictionary to store activation norms for every layer
            self.activation_norm = {}
            for l in range(self.n_layers):
                self.activation_norm[l] = ColumnNormCalculator()
        self.timestep = 0
        self.layer = 0
    
    def update_time_layer(self):
        # updates the timestep when self.layer reaches the last layer
        if self.layer == self.n_layers - 1:
            self.layer = 0
            self.timestep += 1
        else:
            self.layer += 1

    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0
    
    def hook_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in UNet
        '''
        
        args = (1.0,)
        if self.replace_fn == GEGLU:
            # First layer of the FFN
            hidden_states, gate = module.proj(input[0]).chunk(2, dim=-1) # input [2, 4096, 320] weight [320, 1280]
            out = hidden_states * module.gelu(gate)  # [2, 4096, 1280] last dim is the num of neurons
            # input_shape = input[0].shape
            # shape = out.shape
            # Store the input activation to the second layer
            save_gate = out.clone().view(-1, out.shape[-1]).detach().cpu() # [8192, 1280]
            # normalize across the sequence length to avoid inf values
            save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1) # normalize [8192, 1280]
            self.activation_norm.update(save_gate, self.timestep, self.layer)

            # update the time step and layer
            self.update_time_layer()

            return out
        
    def hook_preact_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in UNet
        '''
        
        args = (1.0,)
        if self.replace_fn == GEGLU:
            # First layer of the FFN
            hidden_states, out = module.proj(input[0]).chunk(2, dim=-1) # input [2, 4096, 320] weight [320, 1280]
            out_ = hidden_states * module.gelu(out)  # [2, 4096, 1280] last dim is the num of neurons
            if self.timestep == self.target_t:
                # input_shape = input[0].shape
                # shape = out.shape
                # Store the input activation to the second layer
                save_out = out.clone().detach().cpu()
                shape1 = save_out.shape
                save_out = torch.nn.functional.normalize(save_out, p=2, dim=1)
                shape2 = save_out.shape
                preactivation_values = out.max(dim=1)[0]
                self.activation_norm.update(preactivation_values, self.timestep, self.layer, l2=False)

            # update the time step and layer
            self.update_time_layer()

            return out_
        
    def unet_ffn_1_hook_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in UNet
        '''
        
        args = (1.0,)
        if self.replace_fn == GEGLU:
            # First layer of the FFN
            hidden_states, gate = module.proj(input[0]).chunk(2, dim=-1)
            out = hidden_states * module.gelu(gate)

            # Store the input activation to the second layer
            save_gate = input[0].clone().view(-1, input[0].shape[-1]).detach().cpu()
            # normalize across the sequence length to avoid inf values
            save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
            self.activation_norm.update(save_gate, self.timestep, self.layer)

            # update the time step and layer
            self.update_time_layer()

            return out
        
    def unet_attn_layer(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in UNet
        '''
        save_gate = input[0].clone().detach().cpu()
        save_gate = save_gate.view(-1, save_gate.shape[-1])
        # normalize across the sequence length to avoid inf values
        save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
        self.activation_norm.update(save_gate, self.timestep, self.layer)
        self.update_time_layer()
        return output
        
    def text_hook_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in text encoder (CLIP)
        '''
                
        # First layer of the FFN
        hidden_states = module.fc1(input[0])
        hidden_states = module.activation_fn(hidden_states)

         # Store the input activation to the second layer
        save_gate = hidden_states.clone().detach().cpu()
        save_gate = save_gate.view(-1, hidden_states.shape[-1])
        # normalize across the sequence length to avoid inf values
        save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
        if self.layer < self.n_layers:
            self.activation_norm[self.layer].add_l2_rows(save_gate)
            
        # Output of the second layer of the FFN
        hidden_states = module.fc2(hidden_states)
        # update the time step and layer
        self.update_time_layer()
        return hidden_states
    
    def unet_conv_hook_fn(self, module, input, output):
        save_output = output.clone().detach().cpu()  # input [1, 3, 32, 32] output [1, 128, 32, 32]  weight [128, 3, 3, 3] number of neurons 128
        save_output = save_output.permute(0, 2, 3, 1)
        save_output = save_output.reshape(-1, save_output.shape[-1]) # view
        save_output = torch.nn.functional.normalize(save_output, p=2, dim=1) # [1024, 128]
        self.activation_norm.update(save_output, self.timestep, self.layer)
        self.update_time_layer()
        return output
    
    def unet_preact_hook_fn(self, module, input, output):
        if self.timestep == self.target_t:
            save_output = output.clone().detach().cpu()
            save_output = torch.nn.functional.normalize(save_output, p=2, dim=1)
            preactivation_values = save_output.max(dim=3)[0].max(dim=2)[0]
            self.activation_norm.update(preactivation_values, self.timestep, self.layer, l2=False)
        self.update_time_layer()
        return output
