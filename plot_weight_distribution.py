import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/kctung/Projects/BlockFusion/logs/SIREN_overfitting_mechhand/20251230-182518")
    parser.add_argument("--model_file_name", type=str, default="pure_siren_model_permuted.pt")
    
    args = parser.parse_args()
    loaded_model = torch.load(os.path.join(args.model_dir, args.model_file_name))
    
    net_state_dict = loaded_model["net_state_dict"]
    
    for k, v in net_state_dict.items():
        if k.startswith('0.'):
            print(f"{k}: {v.shape}")
            plt.figure(figsize=(6, 4))
            plt.hist(v.flatten().cpu().numpy(), bins=100)
            # Add title and labels
            plt.title(f"Histogram of SIREN weights at {k}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(args.model_dir, f"{k}.png"))
            plt.close()