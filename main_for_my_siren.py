import os

from dataset import SirenWeightDataset
from hd_utils import Config, get_mlp
from hyperdiffusion_temp import HyperDiffusion

# Using it to make pyrender work on clusters
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
from datetime import datetime
from os.path import join

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from transformer import Transformer

from temp_exps_helper import calculate_siren_weights_n_parameters
from assess_geometry_loss import GeometryLossEvaluator

@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="train_vorts",
)
def main(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    mlp_kwargs = None


    # set the seed for reproducibility
    rand_seed = Config.get("rand_seed")
    pl.seed_everything(rand_seed, workers=True)

    # TODO: can pass my SIREN model config here later
    # so, I don't need to hard code model config
    # mlp_kwargs = Config.config["mlp_config"]["params"]

    # load pre-trained weights
    loaded_model = torch.load(Config.get("siren_path"), map_location="cpu")
    
    if Config.get("mode") == "train":
        # get the hydra log directory
        run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # create tensorboard logger
        tensorboard_writer = TensorBoardLogger(save_dir=run_dir)
    elif Config.get("mode") == "test":
        run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # NOTE: for testing ckpt generated before refactoring the code
        # run_dir = config.run_dir
        #TODO: double check if it's necessary to create tensorboard logger again for evaluation stage
        tensorboard_writer = None

    train_dt = val_dt = test_dt = None

    # Initialize Transformer for HyperDiffusion
    layers, layer_names = calculate_siren_weights_n_parameters(loaded_model)
    print(f"layers: {layers}, layer_names: {layer_names}")
    model = Transformer(
        layers, layer_names, **Config.config["transformer_config"]["params"]
    ).cuda()


    if cfg.transformer_config.params.condition == "light":
        cond_inputs = loaded_model["light_dir_cartesian"]
    elif cfg.transformer_config.params.condition == "volume_timestep":
        cond_inputs = loaded_model["timesteps"]


    train_dt = SirenWeightDataset(
        loaded_model["net_state_dict"],
        cond_inputs,
        model.dims,
        cfg,
        standardize=Config.get("standardize"),
        pre_sampled_coord_groups=loaded_model["pre_sampled_coord_groups"],
        pre_sampled_value_groups=loaded_model["pre_sampled_value_groups"]
    )
    train_dl = DataLoader(
        train_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        drop_last=True,
    )
    
    # TODO: be aware of the batch size passed in (might not work for dist training now)
    geometry_loss_evaluator = GeometryLossEvaluator(train_dt.layer_keys, train_dt.token_shapes, train_dt.token_offsets, Config.get("batch_size"), train_dt.token_means, train_dt.token_stds, Config.get("standardize"))
    # normalize with std of all siren weights (just for experiment)
    # cfg.normalization_factor = (1.0 / (train_dt.std)).item()

    # These two dl's are just placeholders
    # currently, we don't prepare val and test SIREN weights for evaluation
    # TODO: might need to prepare later for better evaluation on unseen lighting directions
    val_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]), batch_size=1, shuffle=False
    )
    test_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]), batch_size=1, shuffle=False
    )

    # print(
    #     "Train dataset length: {} Val dataset length: {} Test dataset length".format(
    #         len(train_dt), len(val_dt), len(test_dt)
    #     )
    # )
    print(
        "Train dataset length: {}".format(
            len(train_dt)
        )
    )
    # dataloader would return a tuple of 4 elements (weights, light directions/volume timesteps, presampled coords, presampled values)
    input_data = next(iter(train_dl))
    print(
        "Input data shape, min, max:",
        input_data[0].shape,
        input_data[0].min(),
        input_data[0].max(),
    )

    best_model_save_path = Config.get("best_model_save_path")
    model_resume_path = Config.get("model_resume_path")

    # Initialize HyperDiffusion
    diffuser = HyperDiffusion(
        model, train_dt, val_dt, test_dt, mlp_kwargs, input_data[0].shape, method, cfg, run_dir, geometry_loss_evaluator
    )

    # # Specify where to save checkpoints
    # just save under the exp directory
    checkpoint_path = run_dir
    
    # best_acc_checkpoint = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="val/1-NN-CD-acc",
    #     mode="min",
    #     dirpath=checkpoint_path,
    #     filename="best-val-nn-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    # )

    # best_mmd_checkpoint = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="val/lgan_mmd-CD",
    #     mode="min",
    #     dirpath=checkpoint_path,
    #     filename="best-val-mmd-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    # )

    # last_model_saver = ModelCheckpoint(
    #     dirpath=checkpoint_path,
    #     filename="last-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    #     save_on_train_epoch_end=True,
    # )
    
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="periodic-{epoch:02d}-{train_loss:.7f}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=Config.get("model_ckpt_freq"),
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=Config.get("epochs"),
        # new version of Pytorch Lightning only support ddp (not dp)
        strategy="ddp",
        logger=tensorboard_writer,
        default_root_dir=checkpoint_path,
        callbacks=[
            periodic_checkpoint,
            lr_monitor
        ],
        check_val_every_n_epoch=Config.get("validation_freq"),
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    if Config.get("mode") == "train":
        # If model_resume_path is provided (i.e., not None), the training will continue from that checkpoint
        trainer.fit(diffuser, train_dl, val_dl, ckpt_path=model_resume_path)
    # best_model_save_path is the path to saved best model
    trainer.test(
        diffuser,
        test_dl,
        ckpt_path=best_model_save_path if Config.get("mode") == "test" else periodic_checkpoint.last_model_path,
    )


if __name__ == "__main__":
    main()
