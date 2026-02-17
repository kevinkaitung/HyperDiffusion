import torch
import argparse
import os
from pathlib import Path

def reshape_weight_back(input, example_network):
    # input should be 1D tensor (for 1 instance)
    layers = dict()
    offset = 0
    for k, v in example_network.items():
        if k.startswith('0.'):
            layers[k] = input[offset: offset+v.numel()].reshape(v.shape)
            offset += v.numel()
    return layers

def reshape_weight_back_all(all_flatten_weights, n_instances, example_network):
    net_state_dict = dict()
    layer_keys = []
    token_offsets = []
    token_shapes = []
    offset = 0
    for k, v in example_network.items():
        if k.startswith('0.'):
            idx_str, layer_name = k.split(".", 1)
            layer_keys.append(layer_name)
            token_offsets.append(offset)
            offset += example_network[k].numel()
            token_shapes.append(example_network[k].shape)
    token_offsets.append(offset)
    
    for idx in range(n_instances):
        for j, layer_key in enumerate(layer_keys):
            net_state_dict[f"{idx}.{layer_key}"] = all_flatten_weights[idx][token_offsets[j]:token_offsets[j + 1]].reshape(token_shapes[j])

    return net_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flatten_weight_file_path", type=str)
    # parser.add_argument("--flatten_weight_file_name", type=str, default="generated_weights_samples.pt")
    # parser.add_argument("--sample_model_path", type=str)

    args=parser.parse_args()

    loaded_model = torch.load(args.flatten_weight_file_path)
    flatten_weight = loaded_model['generated_weights_samples']
    # sample_model = torch.load(args.sample_model_path)
    sample_model = torch.load("/home/kctung/Projects/HyperDiffusion/logs/cond_diff_zebrafish_2000_verify_geo_loss/2026-02-13_00-52-29/sample_siren_7000_test.pt")
    
    ### section to convert only 1 instance
    # use the first sample in the input (flatten weight)
    # reshape_weight = reshape_weight_back(flatten_weight[0], sample_model['net_state_dict'])
    ### section end
    
    ### section to convert all instances
    n_instances = len(flatten_weight)
    assert n_instances == len(loaded_model['light_dir_cartesian'])

    reshape_weight = reshape_weight_back_all(flatten_weight, n_instances, sample_model['net_state_dict'])
    
    save_model = dict()
    save_model['net_state_dict'] = reshape_weight
    # put fake light direction
    # save_model['light_dir_cartesian'] = [[0.5, 0.5, 0.5]]
    # save_model['light_dir_spherical'] = [[0.5, 0.5]]
    # if having real light directions, put them
    # save_model['light_dir_cartesian'] = loaded_model['light_dir_cartesian'][0:1]
    save_model['light_dir_cartesian'] = loaded_model['light_dir_cartesian']
    # save_model['light_dir_spherical'] = [[0.5, 0.5]]
    # import pdb; pdb.set_trace()
    
    path = Path(args.flatten_weight_file_path)
    dir_path = path.parent
    old_file_name = path.name
    
    prefix = "generated_weights_samples_"
    # if passed file name has specific prefix, extract the rest of the name and append to new filename
    if old_file_name.startswith(prefix):
        new_file_name = f"sample_siren_{old_file_name[len(prefix):]}"
    # otherwise, give it a simple name
    else:
        new_file_name = "sample_siren.pt"
    
    torch.save(save_model, os.path.join(dir_path, new_file_name))