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

import wandb
from transformer import Transformer

sys.path.append("siren")

from temp_exps_helper import calculate_siren_weights_n_parameters

@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="train_vorts",
)
def main(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    mlp_kwargs = None


    # My note: is this used for generate mesh?
    # I might not need this
    # # In HyperDiffusion, we need to know the specifications of MLPs that are used for overfitting
    # if "hyper" in method:
    #     mlp_kwargs = Config.config["mlp_config"]["params"]

    # load pre-trained weights
    loaded_model = torch.load(Config.get("siren_path"))
    
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
    
    # wandb.init(
    #     project="hyperdiffusion",
    #     dir=config["tensorboard_log_dir"],
    #     settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    #     tags=[Config.get("mode")],
    #     mode="disabled" if Config.get("disable_wandb") else "online",
    #     config=dict(config),
    # )

    # wandb_logger = WandbLogger()
    # wandb_logger.log_text("config", ["config"], [[str(config)]])
    # print("wandb", wandb.run.name, wandb.run.id)

    train_dt = val_dt = test_dt = None

    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    # My note: in my case, sets of pre-trained weights are in a single .pth file
    mlps_folder_train = Config.get("mlps_folder_train")

    # Initialize Transformer for HyperDiffusion
    if "hyper" in method:
        # mlp = get_mlp(mlp_kwargs)
        # state_dict = mlp.state_dict()
        # layers = []
        # layer_names = []
        # for l in state_dict:
        #     shape = state_dict[l].shape
        #     layers.append(np.prod(shape))
        #     layer_names.append(l)
        # layers and layer_names would be the parameters of hash encoding
        # layers, layer_names = calculate_encoder_n_parameters_by_level(model_config=pretrained_weights["model_state_dict"])
        # layers, layer_names, n_params_mlp = calculate_mlp_n_parameters(
        #     model_config=pretrained_weights["model_state_dict"]
        # )
        # assert n_params_mlp == pretrained_weights["model_state_dict"]["weights0"].shape[0]
        layers, layer_names = calculate_siren_weights_n_parameters(loaded_model)
        print(f"layers: {layers}, layer_names: {layer_names}")
        model = Transformer(
            layers, layer_names, **Config.config["transformer_config"]["params"]
        ).cuda()
    # # Initialize UNet for Voxel baseline
    # else:
    #     model = ldm.ldm.modules.diffusionmodules.openaimodel.UNetModel(
    #         **Config.config["unet_config"]["params"]
    #     ).float()

    # we use our own pre-trained data, no need this section of code
    # dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    # train_object_names = np.genfromtxt(
    #     os.path.join(dataset_path, "train_split.lst"), dtype="str"
    # )
    # if not cfg.mlp_config.params.move:
    #     train_object_names = set([str.split(".")[0] for str in train_object_names])
    # Check if dataset folder already has train,test,val split; create otherwise.
    if method == "hyper_3d":
        # mlps_folder_all = mlps_folder_train
        # all_object_names = np.array(
        #     [obj for obj in os.listdir(dataset_path) if ".lst" not in obj]
        # )
        # total_size = len(all_object_names)
        # val_size = int(total_size * 0.05)
        # test_size = int(total_size * 0.15)
        # train_size = total_size - val_size - test_size
        # if not os.path.exists(os.path.join(dataset_path, "train_split.lst")):
        #     train_idx = np.random.choice(
        #         total_size, train_size + val_size, replace=False
        #     )
        #     test_idx = set(range(total_size)).difference(train_idx)
        #     val_idx = set(np.random.choice(train_idx, val_size, replace=False))
        #     train_idx = set(train_idx).difference(val_idx)
        #     print(
        #         "Generating new partition",
        #         len(train_idx),
        #         train_size,
        #         len(val_idx),
        #         val_size,
        #         len(test_idx),
        #         test_size,
        #     )

        #     # Sanity checking the train, val and test splits
        #     assert len(train_idx.intersection(val_idx.intersection(test_idx))) == 0
        #     assert len(train_idx.union(val_idx.union(test_idx))) == total_size
        #     assert (
        #         len(train_idx) == train_size
        #         and len(val_idx) == val_size
        #         and len(test_idx) == test_size
        #     )

        #     np.savetxt(
        #         os.path.join(dataset_path, "train_split.lst"),
        #         all_object_names[list(train_idx)],
        #         delimiter=" ",
        #         fmt="%s",
        #     )
        #     np.savetxt(
        #         os.path.join(dataset_path, "val_split.lst"),
        #         all_object_names[list(val_idx)],
        #         delimiter=" ",
        #         fmt="%s",
        #     )
        #     np.savetxt(
        #         os.path.join(dataset_path, "test_split.lst"),
        #         all_object_names[list(test_idx)],
        #         delimiter=" ",
        #         fmt="%s",
        #     )

        # val_object_names = np.genfromtxt(
        #     os.path.join(dataset_path, "val_split.lst"), dtype="str"
        # )
        # val_object_names = set([str.split(".")[0] for str in val_object_names])
        # test_object_names = np.genfromtxt(
        #     os.path.join(dataset_path, "test_split.lst"), dtype="str"
        # )
        # test_object_names = set([str.split(".")[0] for str in test_object_names])
        # # assert len(train_object_names) == train_size, f"{len(train_object_names)} {train_size}"

        train_dt = SirenWeightDataset(
            loaded_model["net_state_dict"],
            loaded_model["light_dir_cartesian"],
            model.dims,
            cfg,
            standardize=False
        )
        train_dl = DataLoader(
            train_dt,
            batch_size=Config.get("batch_size"),
            shuffle=True,
        )
        # normalize with std of all siren weights (just for experiment)
        # cfg.normalization_factor = (1.0 / (train_dt.std)).item()
        # val_dt = WeightDataset(
        #     mlps_folder_train,
        #     wandb_logger,
        #     model.dims,
        #     mlp_kwargs,
        #     cfg,
        #     val_object_names,
        # )
        # test_dt = WeightDataset(
        #     mlps_folder_train,
        #     wandb_logger,
        #     model.dims,
        #     mlp_kwargs,
        #     cfg,
        #     test_object_names,
        # )
    # elif method == "raw_3d":
    #     dataset_path = os.path.join(
    #         Config.config["dataset_dir"], Config.config["dataset"]
    #     )
    #     train_dt = VoxelDataset(
    #         dataset_path, wandb_logger, model.dims, mlp_kwargs, cfg, train_object_names
    #     )
    #     train_dl = DataLoader(
    #         train_dt, batch_size=Config.get("batch_size"), shuffle=True, num_workers=2
    #     )

    # These two dl's are just placeholders, during val and test evaluation we are looking at test_split.lst,
    # val_split.lst files, inside calc_metrics methods
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
    # dataloader would return a tuple of 2 elements (weights, light directions)
    input_data = next(iter(train_dl))
    print(
        "Input data shape, min, max:",
        input_data[0].shape,
        input_data[0].min(),
        input_data[0].max(),
    )
    # print(f"Light Direction: {input_data[1]}")

    best_model_save_path = Config.get("best_model_save_path")
    model_resume_path = Config.get("model_resume_path")

    # Initialize HyperDiffusion
    diffuser = HyperDiffusion(
        model, train_dt, val_dt, test_dt, mlp_kwargs, input_data[0].shape, method, cfg, run_dir
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
        # strategy="ddp",
        strategy="dp",
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
    # # best_model_save_path is the path to saved best model
    trainer.test(
        diffuser,
        test_dl,
        ckpt_path=best_model_save_path if Config.get("mode") == "test" else periodic_checkpoint.last_model_path,
    )


if __name__ == "__main__":
    main()
