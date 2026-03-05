# HACK: currently create symbolic link to BlockFusion's simple raymarcher
# TODO: organize the code cleaner between the repos
from simple_raymarcher_with_shadow import *
# from pysampler import decode_shadow, decode, create_sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import distributions as dist
import functorch
from assess_geometry_loss import *

# some design note:
# might pre-generate sample coordinates (for all camera positions) for rendering loss here
# -> sample coordinates should be fixed across all instances (but different across different cam pos)
# -> during runtime, just sample a subset of pixels for quick eval
# GT images should be different across all instances
# -> maybe put with the dataset and return like the presampled values of geometry loss 


# HACK: directly inherit from GeometryLossEvaluator because it already has the logic to handle destandardization
# TODO: should refactor to create another base class to handle common tasks and let geometry/rendering Loss evaluators inherit it
class RenderingLossEvaluator(GeometryLossEvaluator):
    def __init__(self, model_layer_keys, model_layer_shapes, element_offsets, 
                 token_means=None, token_stds=None, is_standardized=False, 
                 camera_configs=None, aabb_configs=None, march_configs=None,
                 raw_data_file_path=None, tfn_file_path=None):
        super().__init__(model_layer_keys, model_layer_shapes, element_offsets,
                         token_means, token_stds, is_standardized)
        
        # get #viewing angles of GT images
        assert len(camera_configs) == len(march_configs), "The numbers of camera and marching configs are mismatched."
        # NOTE: set the #viewing angles here
        # self.n_viewing_angles = len(camera_configs)
        # HACK: try less viewing angles first
        self.n_viewing_angles = 2
        
        # None of the tensor created here would require gradient
        # set no grad to save memory
        with torch.no_grad():
            # prepare ray origins and ray directions of all viewing angles (place on cpu first, move to gpu during training)
            self.ray_origins_groups = []
            self.ray_directions_groups = []
            for idx in range(self.n_viewing_angles):
                ray_origins, ray_directions = generate_rays(camera=Camera(**camera_configs[idx]), device="cpu")
                self.ray_origins_groups.append(ray_origins)
                self.ray_directions_groups.append(ray_directions)
            
            # prepare marching cfg and scene BB
            self.marching_cfg_groups = [MarchConfig(**march_configs[idx]) for idx in range(self.n_viewing_angles)]
            self.scene_aabb_groups = [aabb_configs[idx] for idx in range(self.n_viewing_angles)]
            
            # prepare tfn lookup table
            with open(tfn_file_path, 'r') as f:
                tfn_json = json5.load(f)
            colorControls = tfn_json["view"]["volume"]["transferFunction"]["colorControls"]
            opacityControl = tfn_json["view"]["volume"]["transferFunction"]["opacityControl"]
            self.tfn_lut = build_transfer_function(colorControls, opacityControl, lut_size=1024)
            
            self.n_sampled_pixels_for_each_GT_image = 128
            self.image_height = ray_origins.shape[0]
            self.image_width = ray_origins.shape[1]
            self.total_pixels = self.image_height * self.image_width
            
            # prepare sampler for scalar value queries
            resolution = tfn_json["dataSource"][0]["dimensions"]
            resolution = [resolution["x"], resolution["y"], resolution["z"]]    
            # self.sampler = create_sampler("structuredRegular", "cuda", dims=resolution, dtype="float32", n_channels=1, filename=raw_data_file_path)
            # NOTE: use pure pytorch implementation instead to get rid of sampler dependency
            # ---- read raw file ----
            volume_np = np.fromfile(raw_data_file_path, dtype=np.float32)
            # reshape to 3D
            volume_np = volume_np.reshape((resolution[2], resolution[1], resolution[0]))
            # convert to torch tensor
            volume_tensor = torch.from_numpy(volume_np)
            # normalize volume to 0~1 with in-place operations to save memory usage            
            volume_min = volume_tensor.min()
            volume_max = volume_tensor.max()
            volume_tensor.sub_(volume_min)
            volume_tensor.div_(volume_max - volume_min)
            # create additional dimension for channel (channel dim in this case is 1)
            volume_tensor = volume_tensor.unsqueeze(-1) # (D, H, W, C)

            # tried to pre-generate all sample points beforehand on CPU to save time and GPU memory during training
            # NOTE: might occupy huge amount of CPU memory (i.e., 12GB for 384(H)x384(W)x1024(n_samples))
            # need to think about whether it is still feasible when #viewing angles are large (currently have 20 at most)
            # luckily, the size won't grow when #shadow_instances increase
            self.ray_sampled_pts_groups = []
            self.ray_sampled_pts_inside_groups = []
            for idx in range(self.n_viewing_angles):
                # get the corresponding parameters for this viewing angle
                ray_origins = self.ray_origins_groups[idx]
                ray_directions = self.ray_directions_groups[idx]
                cfg = self.marching_cfg_groups[idx]
                scene_aabb = self.scene_aabb_groups[idx]
                
                # NOTE: the operation would operate on CPU
                # TODO: move to GPU if necessary
                H, W, _ = ray_origins.shape
                device = ray_origins.device
                
                # --- Sample t values along each ray ---
                t_vals = torch.linspace(cfg.t_near, cfg.t_far, cfg.n_samples, device=device)
                # Perturb samples slightly (optional, helps reduce banding)
                if cfg.n_samples > 1:
                    dt = (cfg.t_far - cfg.t_near) / cfg.n_samples
                    noise = torch.rand(H, W, cfg.n_samples, device=device) * dt
                    t_vals = t_vals.unsqueeze(0).unsqueeze(0) + noise   # (H, W, N)
                else:
                    t_vals = t_vals.unsqueeze(0).unsqueeze(0).expand(H, W, -1)

                # --- World-space sample positions ---
                # origins: (H, W, 1, 3),  directions: (H, W, 1, 3),  t: (H, W, N, 1)
                pts = (ray_origins.unsqueeze(2)
                    + ray_directions.unsqueeze(2) * t_vals.unsqueeze(-1))  # (H, W, N, 3)
                
                # --- Map world coords to [0,1] volume space via AABB ---
                aabb_min = scene_aabb[0].to(device)   # (3,)
                aabb_max = scene_aabb[1].to(device)   # (3,)
                
                pts_coords_norm = (pts - aabb_min) / (aabb_max - aabb_min + 1e-8)   # (H, W, N, 3)
                pts_values = sample_volume_trilinear(volume_tensor, pts_coords_norm)  # (H, W, N, 1)
                
                # -- mask: True for points inside the bounding box [0, 1]^3 --
                inside_mask = ((pts_coords_norm >= aabb_min) & (pts_coords_norm <= aabb_max)).all(dim=-1)
                # TODO: probably can be used to filter out those pixels representing the background
                
                # concatenate sampled pts scalar values after sampled pts coords
                pts_coords_values = torch.cat([pts_coords_norm, pts_values], dim=-1)
                
                self.ray_sampled_pts_groups.append(pts_coords_values)
                self.ray_sampled_pts_inside_groups.append(inside_mask)
    
    def single_forward(self, params, x):
        return torch.nn.utils.stateless.functional_call(self.net_template, params, x)

    # TODO: let the dataset return GT images and SIREN_indices!!
    def evaluate_rendering_loss(self, flatten_siren_weights,
                                pre_rendered_GT_images):
        # conceptually, we need to iterate over each siren and each GT images like this
        # for each_siren_ins
        #     for each_GT_image
        #         do ray-marching
        # ray marching operations should be optimized either 
        # 1. parallel network inference across siren instances (vmap for network inference)
        # 2. each GT images (generate ts, ray_origins, ray_directions from different angle at once)
        
        device = flatten_siren_weights.device
        
        N, V, H, W, _ = pre_rendered_GT_images.shape
        assert N == flatten_siren_weights.shape[0]
        assert V == self.n_viewing_angles, "#rendered GT images doesn't match n_viewing_angles"
        
        pre_rendered_GT_images_flat = pre_rendered_GT_images.reshape(N, V, -1, 3)
        
        rendering_loss = torch.tensor(0.0).to(device)
        # iterate through every GT images from different camera positions
        for idx in range(self.n_viewing_angles):
            
            cfg = self.marching_cfg_groups[idx]
            # pre-calculated sampled pts should be on the CPU now
            ray_sampled_pts_flat = self.ray_sampled_pts_groups[idx].reshape(-1, cfg.n_samples, 4) # (H,W,N,4) -> (H*W,N,4)
            ray_sampled_pts_inside_flat = self.ray_sampled_pts_inside_groups[idx].reshape(-1, cfg.n_samples) # (H,W,N) -> (H*W,N)
            
            # randomly select some rays 
            selected_rays_indices = torch.randint(0, self.total_pixels, (self.n_sampled_pixels_for_each_GT_image,))
            
            selected_rays_pts_flat = ray_sampled_pts_flat[selected_rays_indices].to(device) # (n_rays, N, 4)
            selected_rays_inside_flat = ray_sampled_pts_inside_flat[selected_rays_indices].to(device) # (n_rays, N)
            
            rendered_batched_rgb = self.ray_march(selected_rays_pts_flat, selected_rays_inside_flat, cfg, flatten_siren_weights)
            # shape of rendered_batched_rgb: # (n_batch, n_rays, 3)
            
            rendering_loss += F.mse_loss(rendered_batched_rgb, pre_rendered_GT_images_flat[:, idx, selected_rays_indices])
        rendering_loss = rendering_loss / self.n_viewing_angles
        return rendering_loss
    
    def ray_march(
        self,
        ray_sampled_pts:    torch.Tensor,   # (n_rays, N, 4)
        inside_mask:        torch.Tensor,   # (n_rays, N)
        cfg:                MarchConfig,
        flatten_siren_weights: Any,
    ):    
        device = ray_sampled_pts.device
        
        pts_flat = ray_sampled_pts.reshape(-1, 4)   # (n_rays*N, 4)
        inside_mask = inside_mask.flatten()     # (n_rays*N)
        
        # density_flat = torch.zeros([pts_flat.shape[0], 1], device=device)   # (n_rays*N, 1)
        # decode(self.sampler, pts_flat, density_flat)
        
        density_flat = pts_flat[:, 3:]      # (n_rays*N, 1)
        
        # batch inference the network as geometry loss eval does
        batched_params = self.build_batched_params(flatten_siren_weights)
        # NOTE: specify in_dims=(0, None) because we use same set of input batch (pts_flat) for all networks
        batched_shadow_flat = functorch.vmap(self.single_forward, in_dims=(0, None))(batched_params, pts_flat[:, :3])
        # batched_shadow_flat: (batch, n_rays*N, 1) -> each network's output would be stack at the first dim
        n_batch = batched_shadow_flat.shape[0]

        # zero out any outside points that decode might have affected
        # density_flat[~inside_mask] = 0.0
        batched_shadow_flat[:, ~inside_mask]  = 0.0
        
        # del inside_mask
        # torch.cuda.empty_cache()

        density = density_flat.reshape(-1, cfg.n_samples, 1)          # (n_rays, N, 1)
        batched_shadow  = batched_shadow_flat.reshape(n_batch, -1, cfg.n_samples, 1)          # (n_batch, n_rays, N, 1)

        # -- transfer function lookup --
        rgba    = sample_transfer_function(self.tfn_lut, density)    # (n_rays, N, 1, 4)
        rgba    = rgba.squeeze(2)                                    # (n_rays, N, 4)
        rgb     = rgba[..., :3]                                      # (n_rays, N, 3)
        alpha   = rgba[..., 3:]                                       # (n_rays, N, 1)

        # -- opacity correction for actual step size --
        # step_size = (cfg.t_far - cfg.t_near) / cfg.n_samples
        # alpha   = opacity_correction(alpha, step=step_size)        # (H, W, N)

        # -- shadow blending: modulate rgb by shadow coefficient --
        batched_shadow = batched_shadow.clamp(0.0, 1.0)             # (n_batch, n_rays, N, 1)
        ambient = 1.4
        
        # should copy rgb n_batch times
        batched_rgb = rgb.unsqueeze(0).repeat(n_batch, 1, 1, 1)     # (n_batch, n_rays, N, 3)
        batched_rgb = torch.lerp(batched_rgb * ambient,
                            batched_rgb * ambient * batched_shadow,
                            0.9)                   # (n_batch, n_rays, N, 3)
        
        alpha_c = alpha     # (n_rays, N, 1)
        
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones(self.n_sampled_pixels_for_each_GT_image, 1, 1, device=device), # T_0 = 1 (no occlusion yet)
                1.0 - alpha_c + 1e-10                                    # (n_rays, N, 1)
            ], dim=1),
            dim=1
        )[:, :-1, :]                                                  # (n_rays, N, 1) drop the last
        
        weights = transmittance * alpha_c                             # (n_rays, N, 1)

        # -- final compositing --
        weights = weights.unsqueeze(0)                                # (1, n_rays, N, 1)
        rendered_batched_rgb = (weights * batched_rgb).sum(dim=2)          # (n_batch, n_rays, 3)
        
        return rendered_batched_rgb
