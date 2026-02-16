import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import distributions as dist
import functorch

# copy from https://github.com/wilsonCernWq/instant-vnr-pytorch/blob/main/core/networks.py

class SineLayer(nn.Module):
    '''Reference: https://github.com/matthewberger/neurcomp/blob/main/siren.py'''
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super(SineLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))


class SirenResBlock(nn.Module):
    '''Reference: https://github.com/matthewberger/neurcomp/blob/main/siren.py'''
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super(SirenResBlock, self).__init__()

        self.features = features
        self.omega_0 = omega_0

        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, inputs):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1 * inputs))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2 * (inputs + sine_2)

class NeurCompNet(torch.nn.Module):
    def __init__(self, n_input_dims=3, n_output_dims=1, bias=False, n_hidden_layers=8, n_neurons=256, is_residual=True):
        super(NeurCompNet, self).__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_hidden_layers + 2
        self.n_neurons = n_neurons
        self.bias = bias
        self.is_residual = is_residual

        net = []
        for l in range(self.n_layers):
            in_dim  = self.n_input_dims  if l == 0 else self.n_neurons
            out_dim = self.n_output_dims if l == self.n_layers - 1 else self.n_neurons
            is_first = (l==0)
            if l != self.n_layers-1:
                if not self.is_residual:
                    net.append(SineLayer(in_dim, out_dim, bias=True, is_first=is_first))
                else:
                    if is_first:
                        net.append(SineLayer(in_dim, out_dim, bias=True, is_first=is_first))
                    else:
                        net.append(SirenResBlock(in_dim, bias=True, ave_first=(l>1), ave_second=(l==(self.n_layers-2))))
            else:
                final_linear = nn.Linear(in_dim, out_dim)
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / (in_dim)) / 30.0, np.sqrt(6 / (in_dim)) / 30.0)
                net.append(final_linear)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        *S, C = x.size()
        assert C == self.n_input_dims
        x = x.view(-1, self.n_input_dims) * 2 - 1     # to [-1, 1]
        x = self.net(x) * 0.5 + 0.5                   # to [ 0, 1]
        return x.view(*S, self.n_output_dims)

# TODO: fix the issue if the training is distributed
# need to know training batch size on one node first
class GeometryLossEvaluator:
    def __init__(self, model_layer_keys, model_layer_shapes, element_offsets, training_batch_size, 
                 token_means=None, token_stds=None, is_standardized=False):
        #HACK: currently hard code the hyperparameters for NeurCompNet as I only work on this set of config now
        # nets = [
        #     NeurCompNet(n_input_dims=3, n_output_dims=1, bias=False, n_hidden_layers=4, n_neurons=128, is_residual=True)
        #     for _ in range(training_batch_size)]
        # no need to convert nets as nn.ModuleList if performing vmap
        # self.nets = nn.ModuleList(nets)
        self.net_template = NeurCompNet(n_input_dims=3, n_output_dims=1, bias=False, n_hidden_layers=4, n_neurons=128, is_residual=True)
        self.model_layer_keys = model_layer_keys
        self.model_layer_shapes = model_layer_shapes
        self.element_offsets = element_offsets
        
        self.token_means = token_means
        self.token_stds = token_stds
        
        self.is_standardized = is_standardized
        
        # NOTE: freeze the weight for evaluating geometry loss
        # for net in self.nets:
        #     for p in net.parameters():
        #         p.requires_grad = False
        # self.nets.eval()
        
        for p in self.net_template.parameters():
            p.requires_grad = False
        self.net_template.eval()
        
    # NOTE: load_state_dict might break computational graph -> fail to pass gradients back into diffusion model
    # def load_params_to_siren_model(self, flatten_siren_weights):
    #     net_dict = dict()
    #     if self.is_standardized:
    #         for idx_i in range(len(self.nets)):
    #             for idx_j, (key, shape, mean, std) in enumerate(zip(self.model_layer_keys, self.model_layer_shapes, self.token_means, self.token_stds)):
    #                 start = self.element_offsets[idx_j]
    #                 end = self.element_offsets[idx_j + 1]
    #                 # NOTE: we need to destandardize here
    #                 net_dict[f'{idx_i}.{key}'] = flatten_siren_weights[idx_i, start:end].reshape(shape) * std + mean
    #     else:
    #         for idx_i in range(len(self.nets)):
    #             for idx_j, (key, shape) in enumerate(zip(self.model_layer_keys, self.model_layer_shapes)):
    #                 start = self.element_offsets[idx_j]
    #                 end = self.element_offsets[idx_j + 1]
    #                 net_dict[f'{idx_i}.{key}'] = flatten_siren_weights[idx_i, start:end].reshape(shape)
    #     # print(net_dict.keys())
    #     self.nets = self.nets.to(flatten_siren_weights.device)
    #     self.nets.load_state_dict(net_dict)

    # def load_params_to_siren_model(self, flatten_siren_weights):
    #     # net_dict = dict()
    #     self.nets = self.nets.to(flatten_siren_weights.device)
    #     if self.is_standardized:
    #         for idx_i in range(len(self.nets)):
    #             for idx_j, (key, shape, mean, std, name_params_tuple) in enumerate(zip(self.model_layer_keys, self.model_layer_shapes, self.token_means, self.token_stds, self.nets[idx_i].named_parameters())):
    #                 start = self.element_offsets[idx_j]
    #                 end = self.element_offsets[idx_j + 1]
    #                 # NOTE: we need to destandardize here
    #                 # net_dict[f'{idx_i}.{key}'] = flatten_siren_weights[idx_i, start:end].reshape(shape) * std + mean
    #                 name_params_tuple[1].copy_(flatten_siren_weights[idx_i, start:end].reshape(shape) * std + mean)
    #     else:
    #         for idx_i in range(len(self.nets)):
    #             for idx_j, (key, shape, name_params_tuple) in enumerate(zip(self.model_layer_keys, self.model_layer_shapes, self.nets[idx_i].named_parameters())):
    #                 start = self.element_offsets[idx_j]
    #                 end = self.element_offsets[idx_j + 1]
    #                 # net_dict[f'{idx_i}.{key}'] = flatten_siren_weights[idx_i, start:end].reshape(shape)
    #                 name_params_tuple[1].copy_(flatten_siren_weights[idx_i, start:end].reshape(shape))
    #     # print(net_dict.keys())
    #     # self.nets.load_state_dict(net_dict)

    # NOTE: use functional_call instead to inference the SIREN model
    # def evaluate_geometry_loss(self, pre_sampled_coord_groups, pre_sampled_value_groups):
    #     net_device = next(self.nets.parameters()).device
    #     pre_sampled_coord_groups = pre_sampled_coord_groups.to(net_device)
    #     pre_sampled_value_groups = pre_sampled_value_groups.to(net_device)
    #     mse_loss = torch.tensor(0.0).to(net_device)
    #     for batch_idx, net in enumerate(self.nets):
    #         output = net(pre_sampled_coord_groups[batch_idx].float()).float()
    #         # NOTE: make output a 1D tensor which is same as pre-sampled value groups
    #         mse_loss += F.mse_loss(output.flatten(), pre_sampled_value_groups[batch_idx])
        
    #     return mse_loss / len(self.nets)
    
    def build_batched_params(self, flatten_siren_weights):
        """
        Build a batched params dict directly from diffusion output weights.
        Shape of each value: (batch_size, *param_shape)
        This keeps the computational graph alive.
        """
        batched_params = {}
        for idx_j, (key, shape) in enumerate(zip(self.model_layer_keys, self.model_layer_shapes)):
            start = self.element_offsets[idx_j]
            end   = self.element_offsets[idx_j + 1]
            
            # flatten_siren_weights: (batch, total_params)
            # sliced: (batch, param_elements) → (batch, *param_shape)
            w = flatten_siren_weights[:, start:end].reshape(-1, *shape)
            
            if self.is_standardized:
                mean = self.token_means[idx_j]  # make sure these are tensors
                std  = self.token_stds[idx_j]
                w = w * std + mean              # still in graph ✓
            
            batched_params[key] = w             # (batch, *param_shape)
        
        return batched_params



    def evaluate_geometry_loss(self, flatten_siren_weights,
                                pre_sampled_coord_groups,
                                pre_sampled_value_groups):
        model_device = flatten_siren_weights.device
        # set non_blocking to True -> tensors die after out of scope -> hope it can avoid OOM error
        coords = pre_sampled_coord_groups.to(model_device, non_blocking=True).float()   # (batch, N, 3)
        values = pre_sampled_value_groups.to(model_device, non_blocking=True).float()   # (batch, N)

        # Batched params built from diffusion output — graph intact ✓
        batched_params = self.build_batched_params(flatten_siren_weights)

        # Use net_template as the "template" architecture for functional_call
        # vmap will automatically unbatch params along dim=0 per call
        # HACK: because my pytorch is too old, need to install functorch to load vmap
        # and load functional_call with very old namespace (torch.nn.utils.stateless)
        # TODO: upgrade Pytorch later
        def single_forward(params, x):
            return torch.nn.utils.stateless.functional_call(self.net_template, params, x)

        batched_output = functorch.vmap(single_forward)(batched_params, coords)
        # batched_output: (batch, N, 1)
        
        geometry_loss = F.mse_loss(batched_output.flatten(1), values)
        
        return geometry_loss