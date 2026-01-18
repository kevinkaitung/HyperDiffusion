import torch
import argparse
import os

def reshape_weight_back(input, example_network):
    # input should be 1D tensor (for 1 instance)
    layers = dict()
    offset = 0
    for k, v in example_network.items():
        if k.startswith('0.'):
            layers[k] = input[offset: offset+v.numel()].reshape(v.shape)
            offset += v.numel()
    return layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flatten_weight_dir", type=str)
    parser.add_argument("--flatten_weight_file_name", type=str, default="generated_weights_samples.pt")
    # parser.add_argument("--sample_model_path", type=str)

    args=parser.parse_args()

    loaded_model = torch.load(os.path.join(args.flatten_weight_dir, args.flatten_weight_file_name))
    flatten_weight = loaded_model['generated_weights_samples']
    # sample_model = torch.load(args.sample_model_path)
    sample_model = torch.load("/home/kctung/Projects/HyperDiffusion/logs/siren_uncond_diffusion_256/20260106-214220/sample_siren.pt")
    
    # use the first sample in the input (flatten weight)
    reshape_weight = reshape_weight_back(flatten_weight[0], sample_model['net_state_dict'])
    
    save_model = dict()
    save_model['net_state_dict'] = reshape_weight
    # put fake light direction
    # save_model['light_dir_cartesian'] = [[0.5, 0.5, 0.5]]
    # save_model['light_dir_spherical'] = [[0.5, 0.5]]
    # if having real light directions, put them
    save_model['light_dir_cartesian'] = loaded_model['light_dir_cartesian'][0:1]
    # save_model['light_dir_spherical'] = [[0.5, 0.5]]
    # import pdb; pdb.set_trace()
    
    torch.save(save_model, os.path.join(args.flatten_weight_dir, 'sample_siren.pt'))