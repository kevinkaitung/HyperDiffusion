import os
from math import ceil, floor
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

try:
    from augment import random_permute_flat, random_permute_mlp, sorted_permute_mlp
    from hd_utils import generate_mlp_from_weights, get_mlp
    import trimesh
    from trimesh.voxel import creation as vox_creation
    from siren.dataio import anime_read
except ImportError:
    warnings.warn(
        "Cannot import original HyperDiffusion deps. "
        "HyperDiffusion features disabled; SIREN Diffusion still available."
    )

class VoxelDataset(Dataset):    
    def __init__(
        self, mesh_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        self.mesh_folder = mesh_folder
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))

        self.mesh_files = []
        if object_names is None:
            self.mesh_files = [
                file
                for file in list(os.listdir(mesh_folder))
                if file not in ["train_split.lst", "test_split.lst", "val_split.lst"]
            ]
        else:
            for file in list(os.listdir(mesh_folder)):
                if file.split(".")[0] in blacklist and cfg.filter_bad:
                    continue

                if (
                    ("_" in file and file.split("_")[1] in object_names)
                    or file in object_names
                    or file.split(".")[0] in object_names
                ):
                    self.mesh_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.cfg = cfg
        self.vox_folder = self.mesh_folder + "_vox"
        os.makedirs(self.vox_folder, exist_ok=True)

    def __getitem__(self, index):
        dir = self.mesh_files[index]
        path = join(self.mesh_folder, dir)
        resolution = self.cfg.vox_resolution
        voxel_size = 1.9 / (resolution - 1)
        total_time = self.cfg.unet_config.params.image_size
        if self.cfg.mlp_config.params.move:
            folder_name = os.path.basename(path)
            anime_file_path = os.path.join(path, folder_name + ".anime")
            nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_file_path)

            def normalize(obj, v_min, v_max):
                vertices = obj.vertices
                vertices -= np.mean(vertices, axis=0, keepdims=True)
                vertices *= 0.95 / (max(abs(v_min), abs(v_max)))
                obj.vertices = vertices
                return obj

            # total_time = min(nf, total_time)
            vert_datas = []
            v_min, v_max = float("inf"), float("-inf")

            frames = np.linspace(0, nf, total_time, dtype=int, endpoint=False)
            if self.cfg.move_sampling == "first":
                frames = np.linspace(
                    0, min(nf, total_time), total_time, dtype=int, endpoint=False
                )

            for t in frames:
                vert_data_copy = vert_data
                if t > 0:
                    vert_data_copy = vert_data + offset_data[t - 1]
                vert_datas.append(vert_data_copy)
                vert = vert_data_copy - np.mean(vert_data_copy, axis=0, keepdims=True)
                v_min = min(v_min, np.amin(vert))
                v_max = max(v_max, np.amax(vert))
            grids = []
            for vert_data in vert_datas:
                obj = trimesh.Trimesh(vert_data, face_data)
                obj = normalize(obj, v_min, v_max)
                voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                    obj, pitch=voxel_size
                )
                voxel_grid.fill()
                grid = voxel_grid.matrix
                padding_amounts = [
                    (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                    for length in grid.shape
                ]
                grid = np.pad(grid, padding_amounts).astype(np.float32)
                grids.append(grid)
            grid = np.stack(grids)
        else:
            mesh: trimesh.Trimesh = trimesh.load(path)
            coords = np.asarray(mesh.vertices)
            coords = coords - np.mean(coords, axis=0, keepdims=True)
            v_max = np.amax(coords)
            v_min = np.amin(coords)
            coords *= 0.95 / (max(abs(v_min), abs(v_max)))
            mesh.vertices = coords
            voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                mesh, pitch=voxel_size
            )
            voxel_grid.fill()
            grid = voxel_grid.matrix
            padding_amounts = [
                (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                for length in grid.shape
            ]
            grid = np.pad(grid, padding_amounts).astype(np.float32)

        # Convert 0 regions to -1, so that the input is -1 or +1.
        grid[grid == 0] = -1

        grid = torch.tensor(grid).float()

        # Doing some sanity checks for 4D and 3D generations
        if self.cfg.mlp_config.params.move:
            assert (
                grid.shape[0] == total_time
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
                and grid.shape[3] == resolution
            )
            return grid, 0
        else:
            assert (
                grid.shape[0] == resolution
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
            )

        return grid[None, ...], 0

    def __len__(self):
        return len(self.mesh_files)


class WeightDataset(Dataset):
    def __init__(
        self, mlps_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        self.mlps_folder = mlps_folder
        self.condition = cfg.transformer_config.params.condition
        files_list = list(os.listdir(mlps_folder))
        blacklist = {}
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))
        if object_names is None:
            self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        else:
            self.mlp_files = []
            for file in list(os.listdir(mlps_folder)):
                # Excluding black listed shapes
                if cfg.filter_bad and file.split("_")[1] in blacklist:
                    continue
                # Check if file is in corresponding split (train, test, val)
                # In fact, only train split is important here because we don't use test or val MLP weights
                if ("_" in file and (file.split("_")[1] in object_names or (
                        file.split("_")[1] + "_" + file.split("_")[2]) in object_names)) or (file in object_names):
                    self.mlp_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.mlp_kwargs = mlp_kwargs
        if cfg.augment in ["permute", "permute_same", "sort_permute"]:
            self.example_mlp = get_mlp(mlp_kwargs)
        self.cfg = cfg
        if "first_weight_name" in cfg and cfg.first_weight_name is not None:
            self.first_weights = self.get_weights(
                torch.load(os.path.join(self.mlps_folder, cfg.first_weight_name))
            ).float()
        else:
            self.first_weights = torch.tensor([0])

    def get_weights(self, state_dict):
        weights = []
        shapes = []
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights)
        prev_weights = weights.clone()

        # Some augmentation methods are available althougwe don't use them in the main paper
        if self.cfg.augment == "permute":
            weights = random_permute_flat(
                [weights], self.example_mlp, None, random_permute_mlp
            )[0]
        if self.cfg.augment == "sort_permute":
            example_mlp = generate_mlp_from_weights(weights, self.mlp_kwargs)
            weights = random_permute_flat(
                [weights], example_mlp, None, sorted_permute_mlp
            )[0]
        if self.cfg.augment == "permute_same":
            weights = random_permute_flat(
                [weights],
                self.example_mlp,
                int(np.random.random() * self.cfg.augment_amount),
                random_permute_mlp,
            )[0]
        if self.cfg.jitter_augment:
            weights += np.random.uniform(0, 1e-3, size=weights.shape)

        if self.transform:
            weights = self.transform(weights)
        # We also return prev_weights, in case you want to do permutation, we store prev_weights to sanity check later
        return weights, prev_weights

    def __getitem__(self, index):
        file = self.mlp_files[index]
        dir = join(self.mlps_folder, file)
        if os.path.isdir(dir):
            path1 = join(dir, "checkpoints", "model_final.pth")
            path2 = join(dir, "checkpoints", "model_current.pth")
            state_dict = torch.load(path1 if os.path.exists(path1) else path2)
        else:
            state_dict = torch.load(dir, map_location=torch.device("cpu"))

        weights, weights_prev = self.get_weights(state_dict)

        if self.cfg.augment == "inter":
            other_index = np.random.choice(len(self.mlp_files))
            other_dir = join(self.mlps_folder, self.mlp_files[other_index])
            other_state_dict = torch.load(other_dir)
            other_weights, _ = self.get_weights(other_state_dict)
            lerp_alpha = np.random.uniform(
                low=0, high=self.cfg.augment_amount
            )  # Prev: 0.3
            weights = torch.lerp(weights, other_weights, lerp_alpha)

        return weights.float(), weights_prev.float(), weights_prev.float()

    def __len__(self):
        return len(self.mlp_files)

class EncodingWeightDataset(Dataset):
    def __init__(
        self, pretrained_weights_info, wandb_logger, model_dims, cfg
    ):
        self.encoding_weights = {k: pretrained_weights_info[k] for k in pretrained_weights_info.keys() if k.startswith('weights')}
        self.n_params = pretrained_weights_info["tree.n_params"].item()
        self.condition = cfg.transformer_config.params.condition
        
        # original approach would use data augmentation for training,
        # maybe we can consider using it in the future
        self.transform = None
        
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.cfg = cfg

    def __getitem__(self, index):
        if index >= self.n_params:
            # need to raise IndexError to avoid infinite loop
            # when directly enumerating the dataset instead of using DataLoader
            raise IndexError(f"Index {index} out of bounds (n_params={self.n_params})")
        return self.encoding_weights[f"weights{index}"]

    def __len__(self):
        return self.n_params
    
class LatentEncodingWeightDataset(Dataset):
    def __init__(
        self, latent_weights, wandb_logger, model_dims, cfg
    ):
        self.latent_weights = latent_weights
        self.n_params = latent_weights.shape[0]
        self.condition = cfg.transformer_config.params.condition
        
        # original approach would use data augmentation for training,
        # maybe we can consider using it in the future
        self.transform = None
        
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.cfg = cfg

    def __getitem__(self, index):
        return self.latent_weights[index].flatten()

    def __len__(self):
        return self.n_params
    
class SirenWeightDataset(Dataset):
    def __init__(
        self, siren_weights, cond_inputs, model_dims, cfg, standardize=False,
        pre_sampled_coord_groups=None, pre_sampled_value_groups=None,
        pre_cal_GT_images=None
    ):
        # receive the siren_weights as loaded_model['net_state_dict'],
        # which has the keys and values
        
        # use the first instance to capture the layer keys of each module instance
        # TODO: might need to check if there are no keys or no first instance etc.
        layer_keys = []
        token_offsets = []
        token_shapes = []
        offset = 0
        for k in siren_weights.keys():
            if k.startswith('0.'):
                idx_str, layer_name = k.split(".", 1)
                layer_keys.append(layer_name)
                token_offsets.append(offset)
                offset += siren_weights[k].numel()
                token_shapes.append(siren_weights[k].shape)
        token_offsets.append(offset)
        n_layer = len(layer_keys)
        n_instances = int(len(siren_weights.keys()) / n_layer)
        
        # create a unified tensor for all instances    
        # 2D (#instances, flatten weights for all layers)
        self.siren_weights = []
        self.cond_inputs = torch.tensor(cond_inputs) if cond_inputs is not None else None
        
        temp = []
        for idx in range(n_instances):
            for layer_key in layer_keys:
                temp.append(siren_weights[f"{idx}.{layer_key}"].flatten())
            self.siren_weights.append(torch.cat(temp, dim=0))
            temp = []
        self.siren_weights = torch.stack(self.siren_weights, dim=0)
        
        # statistics before any standardization
        self.std = self.siren_weights.std().item()
        self.mean = self.siren_weights.mean().item()
        
        token_means = []
        token_stds = []
        # calculate per-token statistic
        # and standardize siren weights tokens
        for idx in range(len(token_offsets) - 1):
            start = token_offsets[idx]
            end = token_offsets[idx + 1]
            this_token = self.siren_weights[:, start:end]
            # storing as plain python number (by adding .item()), so no need to worry about device of tensors
            token_means.append(this_token.mean().item())
            token_stds.append(this_token.std().item())
            if standardize:
                self.siren_weights[:, start:end] = (this_token - this_token.mean()) / (this_token.std() + 0.0000000001)
        
        self.standardize = standardize
        self.token_means = token_means
        self.token_stds = token_stds
        self.token_offsets = token_offsets
        self.layer_keys = layer_keys
        self.token_shapes = token_shapes
        
        self.n_params = n_instances
        self.condition = cfg.transformer_config.params.condition
        
        # original approach would use data augmentation for training,
        # maybe we can consider using it in the future
        self.transform = None
        
        self.model_dims = model_dims
        self.cfg = cfg
        
        # prepare the presample points for evaluating geometry loss (each instance should have a set of coords/scalar values pair)
        if (pre_sampled_coord_groups is not None) & (pre_sampled_value_groups is not None):
            self.pre_sampled_coord_groups = pre_sampled_coord_groups
            self.pre_sampled_value_groups = pre_sampled_value_groups
        else:
            # the case that geometry loss is not enabled
            self.pre_sampled_coord_groups = [torch.empty(0) for _ in range(n_instances)]
            self.pre_sampled_value_groups = [torch.empty(0) for _ in range(n_instances)]    
        self.pre_sampled_batch_size = 2**12
        
        # self.pre_cal_GT_images = pre_cal_GT_images
        # HACK: try get less GT images for initial training
        if pre_cal_GT_images is not None:
            n_GT_imgs = 2
            self.pre_cal_GT_images = [pre_cal_GT_images[idx][:n_GT_imgs] for idx in range(n_instances)]
        else:
            # the case that rendering loss is not enabled
            self.pre_cal_GT_images = [torch.empty(0) for _ in range(n_instances)]

    def __getitem__(self, index):
        pre_sampled_coord = self.pre_sampled_coord_groups[index]
        if pre_sampled_coord.numel() > 0:
            selected_sampled_indices = torch.randint(0, pre_sampled_coord.shape[0], (self.pre_sampled_batch_size,), device=pre_sampled_coord.device)
            coords = self.pre_sampled_coord_groups[index][selected_sampled_indices]
            values = self.pre_sampled_value_groups[index][selected_sampled_indices]
        else:
            # the case that geometry loss is not enabled
            coords = pre_sampled_coord                          # empty (0)
            values = self.pre_sampled_value_groups[index]       # empty (0)
        return self.siren_weights[index].flatten(), self.cond_inputs[index], coords, values, self.pre_cal_GT_images[index]

    def __len__(self):
        return self.n_params
    
    def get_all_cond_inputs(self):
        return self.cond_inputs
    
# only used for testing_step
# TODO: might want to extend to also recieve test set SIREN weights (if given) and consolidate with SirenWeightDataset
# possibly for evaluation in the test step (either directly compare SIREN weights themselves or decode back to volume)
class TestsetDataset(Dataset):
    def __init__(
        self, cond_inputs
    ):
        self.cond_inputs = torch.tensor(cond_inputs)
        self.n_params = len(self.cond_inputs)
    
    def __getitem__(self, index):
        return self.cond_inputs[index]
    
    def __len__(self):
        return self.n_params
    
class TemporalSirenWeightDataset(SirenWeightDataset):
    def __init__(
        self, siren_weights, cond_inputs, model_dims, cfg, standardize=False, pre_sampled_coord_groups=None, pre_sampled_value_groups=None
    ):
        super().__init__(siren_weights, None, model_dims, cfg, standardize, pre_sampled_coord_groups, pre_sampled_value_groups)
        self.temporal_indices = cond_inputs
        first_frame_cond_strategy = "self"
        
        # Overwrite self.cond_inputs using already-processed siren_weights
        # No extra memory — just stacking views into the same tensor
        prev_weights = []
        for idx in self.temporal_indices:
            # first frame as edge case:
            if idx == 0:
                if first_frame_cond_strategy == "zeros":
                    prev_weights.append(torch.zeros(self.siren_weights.shape[1]))
                elif first_frame_cond_strategy == "self":
                    prev_weights.append(self.siren_weights[0].flatten())
            else:
                prev_weights.append(self.siren_weights[idx - 1].flatten())
        
        self.cond_inputs = torch.stack(prev_weights, dim=0)  # (n_instances, weight_dim)
        self.temporal_indices = torch.tensor(self.temporal_indices)
    
    def get_all_temporal_indices(self):
        return self.temporal_indices