import torch
import torch.nn as nn
import numpy as np

def extract_weights_at_level(model_config: dict, level: int, chunk_size: int):
    """
    Helper function to extract the weights at a specific level.
    """
    # because these configuration numbers are stored as tensors,
    # we need to convert them to integers
    n_levels = model_config['configuration.n_levels'].item()
    n_features_per_level = model_config['configuration.n_features_per_level'].item()
    base_resolution = model_config['configuration.base_resolution'].item()
    log2_hashmap_size = model_config['configuration.log2_hashmap_size'].item()
    per_level_scale = model_config['configuration.per_level_scale'].item()
    n_input_dims = 3
    
    n_params = model_config['tree.n_params'].item()
    
    assert level < n_levels, f"Level {level} is out of bounds for {n_levels} levels."
    
    offset = 0
    length = 0
    for i in range(level + 1):
        offset += length
        scale = grid_scale(i, per_level_scale, base_resolution)
        resolution = grid_resolution(scale)
        length = resolution ** n_input_dims
        # Make sure memory accesses will be aligned
        # should align with the calculation in tiny-cuda-nn
        length = (length + 8 - 1) // 8 * 8
        length = min(length, 1 << log2_hashmap_size)
        length *= n_features_per_level
    
    # decompose the grid into chunks and reorder elements in each grid to be sequentially stored
    for i in range(n_params):
        this_weights = model_config[f"weights{i}"]
        
        temp = this_weights[offset:offset+length]
        temp = temp.reshape(resolution, resolution, resolution, n_features_per_level)
        temp = temp.unfold(0, chunk_size, chunk_size).unfold(1, chunk_size, chunk_size).unfold(2, chunk_size, chunk_size)
        temp = temp.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
        this_weights = temp.flatten()

        model_config[f"weights{i}"] = this_weights
    
    
    n_params_by_layer = []
    names_by_layer = []
    # calculate layers
    scale = grid_scale(level, per_level_scale, base_resolution)
    resolution = grid_resolution(scale)
    for i in range((resolution) // chunk_size):
        for j in range(resolution // chunk_size):
            for k in range(resolution // chunk_size):
                n_params_by_layer.append(chunk_size * chunk_size * chunk_size * n_features_per_level)
                names_by_layer.append(f"level_{level}_layer_{i}_{j}_{k}")
    
    return  n_params_by_layer, names_by_layer, model_config


def chunk_encoder_dense_grid(model_config: dict, chunk_size: int):
    """
    Helper function to chunk the encoder dense grid.
    """
    # because these configuration numbers are stored as tensors,
    # we need to convert them to integers
    n_levels = model_config['configuration.n_levels'].item()
    n_features_per_level = model_config['configuration.n_features_per_level'].item()
    base_resolution = model_config['configuration.base_resolution'].item()
    log2_hashmap_size = model_config['configuration.log2_hashmap_size'].item()
    per_level_scale = model_config['configuration.per_level_scale'].item()
    n_input_dims = 3
    
    n_params = model_config['tree.n_params'].item()
    
    # decompose the grid into chunks and reorder elements in each grid to be sequentially stored
    for i in range(n_params):
        this_weights = model_config[f"weights{i}"]
        prev_offset = 0
        for j in range(n_levels):
            scale = grid_scale(j, per_level_scale, base_resolution)
            resolution = grid_resolution(scale)
            length = resolution ** n_input_dims
            # Make sure memory accesses will be aligned
            # should align with the calculation in tiny-cuda-nn
            length = (length + 8 - 1) // 8 * 8
            length = min(length, 1 << log2_hashmap_size)
            length *= n_features_per_level
            
            temp = this_weights[prev_offset:prev_offset + length]
            temp = temp.reshape(resolution, resolution, resolution, n_features_per_level)
            temp = temp.unfold(0, chunk_size, chunk_size).unfold(1, chunk_size, chunk_size).unfold(2, chunk_size, chunk_size)
            temp = temp.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
            this_weights[prev_offset:prev_offset + length] = temp.flatten()
            prev_offset += length

        model_config[f"weights{i}"] = this_weights
        
    n_params_by_layer = []
    names_by_layer = []
    
    # calculate layers
    for m in range(n_levels):
        scale = grid_scale(m, per_level_scale, base_resolution)
        resolution = grid_resolution(scale)
        for i in range((resolution) // chunk_size):
            for j in range(resolution // chunk_size):
                for k in range(resolution // chunk_size):
                    n_params_by_layer.append(chunk_size * chunk_size * chunk_size * n_features_per_level)
                    names_by_layer.append(f"level_{m}_layer_{i}_{j}_{k}")
    return n_params_by_layer, names_by_layer, model_config


def calculate_encoder_n_parameters_by_level(model_config: dict):
    """
    Helper function to calculate the number of parameters by level.
    """
    # because these configuration numbers are stored as tensors,
    # we need to convert them to integers
    n_levels = model_config['configuration.n_levels'].item()
    n_features_per_level = model_config['configuration.n_features_per_level'].item()
    base_resolution = model_config['configuration.base_resolution'].item()
    log2_hashmap_size = model_config['configuration.log2_hashmap_size'].item()
    per_level_scale = model_config['configuration.per_level_scale'].item()
    n_input_dims = 4

    # offsets_by_level = []
    # offset = 0
    n_params_by_level = []
    names_by_level = []
    for i in range(n_levels):
        scale = grid_scale(i, per_level_scale, base_resolution)
        resolution = grid_resolution(scale)
        length = resolution ** n_input_dims
        # Make sure memory accesses will be aligned
        # should align with the calculation in tiny-cuda-nn
        length = (length + 8 - 1) // 8 * 8
        length = min(length, 1 << log2_hashmap_size)
        length *= n_features_per_level
        # offsets_by_level.append(offset)
        n_params_by_level.append(length)
        names_by_level.append(f"level_{i}")
        # offset += length

    return n_params_by_level, names_by_level

@torch.no_grad()
def grid_scale(level:int, per_level_scale:float, base_resolution:float):
	return np.power(np.float32(2), np.float32(level) * np.log2(np.float32(per_level_scale))) * np.float32(base_resolution) - np.float32(1.0)
	
@torch.no_grad()
def grid_resolution(scale:float):
	return np.int32(np.ceil(np.float32(scale))) + 1

def calculate_mlp_n_parameters(model_config: dict):
    """
    Helper function to calculate the number of parameters in the MLP.
    """
    n_input_dims = model_config['configuration.n_levels'].item() * model_config['configuration.n_features_per_level'].item()
    n_neurons = model_config['configuration.n_neurons'].item()
    n_hidden_layers = model_config['configuration.n_hidden_layers'].item()
    
    n_params_by_layer = []
    names_by_layer = []
    
    # Calculate the number of parameters in the MLP
    n_params_mlp = 0
    for i in range(n_hidden_layers + 1):
        if i == 0:
            in_dim = n_input_dims
        else:
            in_dim = n_neurons
        if i == n_hidden_layers:
            # tcnn might padd the last layer to 16 (instead of 1 dimension)
            out_dim = 16
        else:
            out_dim = n_neurons
        n_params_mlp += in_dim * out_dim
        n_params_by_layer.append(in_dim * out_dim)
        names_by_layer.append(f"layer_{i}")

    return n_params_by_layer, names_by_layer, n_params_mlp

def calculate_latent_weights_n_parameters(data_config: dict):
    latent_space_num_chs = data_config["z_shape"][0]
    latent_space_3D_dims = data_config["z_shape"][1:]
    
    n_params_by_layer = []
    names_by_layer = []
    
    # Calculate the number of parameters in the latent space
    for i in range(latent_space_num_chs):
        n_params_by_layer.append(latent_space_3D_dims[0] * latent_space_3D_dims[1] * latent_space_3D_dims[2])
        names_by_layer.append(f"channel_{i}")
    
    return n_params_by_layer, names_by_layer

def calculate_siren_weights_n_parameters(data_config: dict):
    siren_weights_state_dict = data_config['net_state_dict']
    
    n_params_by_layer = []
    names_by_layer = []
    
    # use the first instance to capture the layer keys of each module instance
    # TODO: might need to check if there are no keys or no first instance etc.
    
    for k in siren_weights_state_dict.keys():
        if k.startswith('0.'):
            n_params_by_layer.append(siren_weights_state_dict[k].numel())
            names_by_layer.append(k.split('.', 1)[1])
            
    return n_params_by_layer, names_by_layer