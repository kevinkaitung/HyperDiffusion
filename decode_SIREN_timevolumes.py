import torch
import argparse
from assess_geometry_loss import NeurCompNet
import numpy as np

def generate_coords_chunks(data_res, chunk_size, device='cpu'):
    """Yield chunks of coordinates from the full 3D grid."""
    gridz, gridy, gridx = torch.meshgrid(
        torch.linspace(0, 1, data_res[2]),  # z slowest
        torch.linspace(0, 1, data_res[1]),
        torch.linspace(0, 1, data_res[0]),  # x fastest
        indexing='ij'
    )
    # the accessing pattern in flattened volume: [1,0,0], [2,0,0], [3,0,0] ... (x change fastest)
    coords = torch.stack([gridx, gridy, gridz], dim=3).reshape(-1, 3)  # [N, 3]
    
    for start in range(0, coords.shape[0], chunk_size):
        end = start + chunk_size
        # allocate memory on CPU, only move to GPU when used for model inference
        yield coords[start:end].to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--SIREN_file_path', type=str, default="../VAE_Reconstructed_triplane.pt")
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--new_filename', type=str, default="diffusion_timevolumes")
    parser.add_argument('--indices_to_decode', type=int, nargs='+', default=[0], help="which indices (in SIREN file) to decode")
    
    args = parser.parse_args()
    
    loaded_model = torch.load(args.SIREN_file_path, map_location="cuda")
    
    indices_to_decode = args.indices_to_decode
    
    all_timesteps = loaded_model['timesteps']
    n_instances = len(all_timesteps)
    
    nets = [
        NeurCompNet(n_input_dims=3, n_output_dims=1, bias=False, n_hidden_layers=4, n_neurons=128, is_residual=True).cuda()
        for _ in range(n_instances)]
    nets = torch.nn.ModuleList(nets)
    nets.load_state_dict(loaded_model['net_state_dict'])
    
    chunk_size = 16384*1024
    data_res = args.dims
    
    with torch.no_grad():
        for batch_idx in range(len(indices_to_decode)):
            preds = []
            print(f"processing batch {batch_idx}...")
            idx = 0
            for coord_chunk in generate_coords_chunks(data_res, chunk_size, device='cuda'):
                # preds.append(net(triplane[batch_idx](coord_chunk, 0)).cpu())
                # need to clamp the value range in case extreme outliers would make the rest of most values gather in one bin
                # preds.append(nets[batch_idx](coord_chunk).clamp(-1.0, 2.0).cpu())
                preds.append(nets[indices_to_decode[batch_idx]](coord_chunk).cpu())
                # used when transforming the input values with inverse sigmoid
                # preds.append(torch.sigmoid(net(triplane[batch_idx](coord_chunk, 0))).cpu())
                print(f"coord chunk: {idx} / memory allocated: {torch.cuda.memory_allocated() / 1024**3} / max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3}")
                idx += 1
            
            # outputs = net(triplane[batch_idx](coords, 0))
            # outputs = outputs.view(raw_data.shape)
            outputs = torch.cat(preds, dim=0)
            outputs.detach().cpu().numpy().astype(np.float32).tofile(f"{args.new_filename}_at_{all_timesteps[indices_to_decode[batch_idx]]}_ins_{indices_to_decode[batch_idx]}.bin")
            del outputs
            torch.cuda.empty_cache()