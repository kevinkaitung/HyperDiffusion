import copy
import os

import numpy as np
import pytorch_lightning as pl
import torch
# deprecated in newer pytorch lightning
# from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tqdm import tqdm

from diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)
from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights)

import matplotlib.pyplot as plt

class HyperDiffusion(pl.LightningModule):
    def __init__(
        self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, cfg, run_dir,
        geometry_loss_evaluator, rendering_loss_evaluator
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.method = method
        self.mlp_kwargs = mlp_kwargs
        self.val_dt = val_dt
        self.train_dt = train_dt
        self.test_dt = test_dt
        self.ae_model = None
        self.run_dir = run_dir
        self.sample_count = min(
            8, Config.get("batch_size")
        )  # it shouldn't be more than 36 limited by batch_size
        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        print("encoded_outs.shape", encoded_outs.shape)
        timesteps = Config.config["timesteps"]
        betas = torch.tensor(np.linspace(1e-4, 2e-2, timesteps))
        self.image_size = encoded_outs[:1].shape

        # Initialize diffusion utiities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[cfg.diff_config.params.model_mean_type],
            model_var_type=ModelVarType[cfg.diff_config.params.model_var_type],
            loss_type=LossType[cfg.diff_config.params.loss_type],
            diff_pl_module=self,
        )
        self.num_samples_for_val = 16
        self.noise_for_val = torch.randn((self.num_samples_for_val, *self.image_size[1:]))
        
        all_train_set_cond_inputs = self.train_dt.get_all_cond_inputs()
        self.selected_idx_for_val = torch.randperm(all_train_set_cond_inputs.shape[0])[:self.num_samples_for_val]
        # or, give a fix set of indices for validation
        # self.selected_idx_for_val = torch.tensor([0, 18, 50, 65, 88, 150, 180, 210, 257, 298, 314, 361, 405, 489, 502, 556])
        self.cond_input_for_val = all_train_set_cond_inputs[self.selected_idx_for_val]
        
        # select some cond inputs for testing step
        # would only be retrieved when test set cond inputs file not provided
        self.num_samples_for_test = 16
        self.selected_idx_for_test = torch.randperm(all_train_set_cond_inputs.shape[0])[:self.num_samples_for_test]
        # or, give a fix set of indices for validation
        # self.selected_idx_for_test = torch.tensor([0, 18, 50, 65, 88, 150, 180, 210, 257, 298, 314, 361, 405, 489, 502, 556])
        self.cond_input_for_test = all_train_set_cond_inputs[self.selected_idx_for_test]
        
        self.geometry_loss_evaluator = geometry_loss_evaluator
        self.rendering_loss_evaluator = rendering_loss_evaluator
        self.condition_type = cfg.transformer_config.params.condition

    def forward(self, images, cond_input):
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(images.shape[0],))
            .long()
            .to(self.device)
        )
        images = images * self.cfg.normalization_factor
        x_t, e = self.diff.q_sample(images, t)
        x_t = x_t.float()
        e = e.float()
        return self.model(x_t, t, cond_input), e

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=Config.get("lr"))
        if self.cfg.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.cfg.scheduler_step, gamma=0.9
            )
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # pytorch lightning would move train_batch to target device automatically (e.g., GPU)
        # so at this point, the input_data is already on the target device (No need to move manually!)
        input_data = train_batch
        # At the first step output first element in the dataset as a sanit check
        if self.trainer.global_step == 0:
            print("Input images shape:", input_data[0].shape)
            print("Conditional inputs shape:", input_data[1].shape)
            print("Presampled coords shape:", input_data[2].shape)
            print("Presampled values shape:", input_data[3].shape)
            print("Precalculated GT images shape:", input_data[4].shape)
        
        # Output statistics every 100 step
        if self.trainer.global_step % 100 == 0:
            print(input_data[0].shape)
            print(
                "Orig weights[0].stats",
                input_data[0].min().item(),
                input_data[0].max().item(),
                input_data[0].mean().item(),
                input_data[0].std().item(),
            )

        # Sample a diffusion timestep
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(input_data[0].shape[0],))
            .long()
            .to(self.device)
        )
        # pass conditioning via model_kwargs (not mlp_kwargs)
        model_kwargs = {
            "cond_input": input_data[1]
        }
        additional_args = {
            "pre_sampled_coord_groups": input_data[2],
            "pre_sampled_value_groups": input_data[3],
            "pre_cal_GT_images": input_data[4]
        }
        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            input_data[0] * self.cfg.normalization_factor,
            t,
            self.mlp_kwargs,
            self.logger,
            model_kwargs=model_kwargs,
            additional_args=additional_args,
            geometry_loss_evaluator=self.geometry_loss_evaluator,
            rendering_loss_evaluator=self.rendering_loss_evaluator
        )
        loss_mse = loss_terms["loss"].mean()
        self.log("train_loss", loss_mse)
        self.log("mse_loss", loss_terms["mse"].mean())
        # Output cosine similarity every 100 step
        if self.trainer.global_step % 100 == 0:
            print("cosine similarity between predicted weights and original weights: ", loss_terms["cos_sim_mean"].mean())
            print("geometry loss: ", loss_terms["geometry_loss"].mean())
            print("rendering loss: ", loss_terms["rendering_loss"].mean())
            print("mse loss: ", loss_terms["mse"].mean())
        self.log("cosine_similarity", loss_terms["cos_sim_mean"].mean())
        self.log("geometry_loss", loss_terms["geometry_loss"].mean())
        self.log("rendering_loss", loss_terms["rendering_loss"].mean())

        loss = loss_mse
        self.log("epoch_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        mean_PSNR, mean_cosine_similarity = self.generate_samples(self.num_samples_for_val, self.noise_for_val, self.cond_input_for_val, False)
        self.log("val/PSNR", mean_PSNR)
        self.log("val/cosine_similarity", mean_cosine_similarity)
    
    # deprecated in newer pytorch lightning
    # def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # epoch_loss = sum(output["loss"] for output in outputs) / len(outputs)
        # self.log("epoch_loss", epoch_loss)


    def print_summary(self, flat, func):
        var = func(flat, dim=0)
        print(
            var.shape,
            var.mean().item(),
            var.std().item(),
            var.min().item(),
            var.max().item(),
        )
        print(var.shape, func(flat))

    
    # only calculate 1 sample
    def cal_and_plot_cosine_similarity_against_gt(self, generate_weight_1_sample, is_test=False):
        '''
        generate_weight_1_sample: 1D tensor (length: length of the all siren weights in one sample)
        '''
        PSNR_list = []
        cosine_similarity_list = []
        for idx, (GT_siren_weight, light_dir, _, _, _) in enumerate(self.train_dt):
            # NOTE: GT_siren_weight should be on CPU, so move to the same device as generate_weight_1_sample
            GT_siren_weight = GT_siren_weight.to(generate_weight_1_sample.device)
            # both 'generate_weight_1_sample' and 'GT_siren_weight' should be flattened SIREN weight of 1 sample (1D tensor)
            mse = torch.nn.functional.mse_loss(generate_weight_1_sample, GT_siren_weight)
            max = GT_siren_weight.max() - GT_siren_weight.min()
            cosine_similarity = torch.nn.functional.cosine_similarity(generate_weight_1_sample, GT_siren_weight, dim=0)
            PSNR = 20 * torch.log10(max / torch.sqrt(mse))
            # print(f"instance {idx}: PSNR {PSNR} / cosine similarity {cosine_similarity}")
            PSNR_list.append(PSNR.item())
            cosine_similarity_list.append(cosine_similarity.item())
            
        plt.plot(range(len(self.train_dt)), PSNR_list)
        plt.xlabel('instance index')
        plt.ylabel('PSNR')
        plt.title(f'Gen SIREN weights quality against all GT SIREN weights in train set (All layers)')
        if is_test:
            save_name = f'gen_weights_PSNR_all_tokens_{self.current_epoch}_test.png'
        else:
            save_name = f'gen_weights_PSNR_all_tokens_{self.current_epoch}.png'    
        plt.savefig(os.path.join(self.run_dir, save_name))
        plt.close()

        plt.plot(range(len(self.train_dt)), cosine_similarity_list)
        plt.xlabel('instance index')
        plt.ylabel('cosine_similarity')
        plt.title(f'Gen SIREN weights quality against all GT SIREN weights in train set (All layers)')
        if is_test:
            save_name = f'gen_weights_cosine_similarity_all_tokens_{self.current_epoch}_test.png'
        else:
            save_name = f'gen_weights_cosine_similarity_all_tokens_{self.current_epoch}.png'    
        plt.savefig(os.path.join(self.run_dir, save_name))
        plt.close()
        
        return np.mean(PSNR_list), np.mean(cosine_similarity_list)
        
    def evaluate_recon_volume_quality(self):
        # TODO: decode the volume with gen samples
        return
    
    def calculate_stats_of_train_set_data(self):
        x_0s = []
        for i, (img, light_dir, _, _, _) in enumerate(self.train_dt):
            x_0s.append(img)
        x_0s = torch.stack(x_0s).to(self.device)
        flat = x_0s.view(len(x_0s), -1)
        # return
        print(x_0s.shape, flat.shape)
        print("Variance With zero-padding")
        self.print_summary(flat, torch.var)
        print("Variance Without zero-padding")
        self.print_summary(flat[:, : Config.get("curr_weights")], torch.var)

        print("Mean With zero-padding")
        self.print_summary(flat, torch.mean)
        print("Mean Without zero-padding")
        self.print_summary(flat[:, : Config.get("curr_weights")], torch.mean)

        stdev = x_0s.flatten().std(unbiased=True).item()
        oai_coeff = (
            0.538 / stdev
        )  # 0.538 is the variance of ImageNet pixels scaled to [-1, 1]
        print(f"Standard Deviation: {stdev}")
        print(f"OpenAI Coefficient: {oai_coeff}")
        
    
    def generate_samples(self, num_samples, noise=None, cond_input=None, is_test=False):
        if noise is not None:
            assert noise.shape[0] == num_samples, (
                "the first dim of noise (batch_size) should match num_samples"
            )
            # NOTE: need to manually move noise onto device
            noise = noise.to(self.device)
        

        self.calculate_stats_of_train_set_data()
        
        # inference with provided conditional input
        model_kwargs = {
            # NOTE: need to manually move conditional inputs onto device
            # since we don't get them from pytroch lightning's validation_step and test_step (where it would move automatically) 
            "cond_input": cond_input.to(self.device)
        }
        # Then, sampling some new shapes -> outputting and rendering them
        x_0s = self.diff.ddim_sample_loop(
            self.model, (num_samples, *self.image_size[1:]), noise=noise, clip_denoised=False, model_kwargs=model_kwargs
        )
        ### section to destandardize
        if self.train_dt.standardize:
            token_means = self.train_dt.token_means
            token_stds = self.train_dt.token_stds
            token_offsets = self.train_dt.token_offsets
            idx = 0
            for token_mean, token_std in zip(token_means, token_stds):
                start = token_offsets[idx]
                end = token_offsets[idx + 1]
                x_0s[:, start:end] = x_0s[:, start:end] * (token_std + 0.0000000001) + token_mean

                idx += 1
        ### end section
        x_0s = x_0s / self.cfg.normalization_factor

        print(
            "x_0s[0].stats",
            x_0s.min().item(),
            x_0s.max().item(),
            x_0s.mean().item(),
            x_0s.std().item(),
        )
        out_pc_imgs = []
        
        # only use the first sample to plot cosine similarity evaluation
        mean_PSNR, mean_cosine_similarity = self.cal_and_plot_cosine_similarity_against_gt(x_0s[0], is_test)
        
        # Save generated weights samples to disk
        if is_test:
            # NOTE: it seems newer pytorch lightning (On sophia) would not set current_epoch to epoch in ckpt file
            # so, it would show epoch 0 here
            save_dir = f"{self.run_dir}/generated_weights_samples_{self.current_epoch}_test.pt"
            selected_idx = self.selected_idx_for_test
        else:
            save_dir = f"{self.run_dir}/generated_weights_samples_{self.current_epoch}_validation.pt"
            selected_idx = self.selected_idx_for_val
        # os.makedirs(save_dir, exist_ok=True)
        if self.condition_type == "light":
            torch.save({"generated_weights_samples": x_0s, "light_dir_cartesian": cond_input.tolist()}, save_dir)
        elif self.condition_type == "volume_timestep":
            torch.save({"generated_weights_samples": x_0s, "timesteps": cond_input.tolist()}, save_dir)
        elif self.condition_type == "prev_volume_weight":
            # TODO: see if I really need to store the actual cond input
            # HACK: a little bit hacky to access selected_idx_for_val for querying temporal indices
            # see how to improve later
            torch.save({"generated_weights_samples": x_0s, "timesteps": self.train_dt.get_all_temporal_indices()[selected_idx].tolist()}, save_dir)
        return mean_PSNR, mean_cosine_similarity

    def test_step(self, test_batch, batch_idx):
        # self.generate_samples(16, noise=self.noise_for_val, cond_input=self.cond_input_for_val, is_test=True)
        # TODO: need to revise for the case that multiple batches would be used in test_step
        self.generate_samples(len(test_batch), cond_input=test_batch, is_test=True)
