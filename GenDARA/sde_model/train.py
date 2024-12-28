import argparse
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import pandas as pd

from model import SdeTrainer
from dataset import SdeDataModule, SdeDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # add argumense to the script
    parser = argparse.ArgumentParser(description='Train the SDE model on the participant augmented RIR dataset')
    parser.add_argument('--gpus', type=int, nargs='+', default=[6], help='List of GPUs to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generators')
    parser.add_argument('--deterministic', type=bool, help='Whether to set the random number generators to be deterministic')
    parser.add_argument('--checkpoint', type=str, default='baseline.ckpt', help='Path to a checkpoint to load the model from. Default is the baseline model')

    args = parser.parse_args()
    print(args)
    
    ########### Training variables and params ###########
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    path_audios = os.path.join(data_dir, 'augmented_rirs')
    path_speech = os.path.join(data_dir, 'VCTK-Corpus' )
    path_dataset_splits = data_dir
    
    config = {
    "max_epochs": 2, #50
    "batch_size": 16,
    "lr": 0.001,
    "sampling_frequency": 32000,
    "duration": 10,
    "kernels": "freq",
    "n_grus": 2,
    "features_set": "all",
    "att_conf": "onSpec"
    }
    
    dataloader_config = {
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True,
        "shuffle": True
    }

    ##### setup #####
    n_gru = config['n_grus']
    kernel = config['kernels']
    features = config['features_set']
    att_conf = config['att_conf']
    
    dfs_paths = {"train": os.path.join(path_dataset_splits, "meta_train.csv"),
                "val": os.path.join(path_dataset_splits, "meta_val.csv"),
                "test": os.path.join(path_dataset_splits, "meta_val.csv"),
    }

    path_speech = { "train": os.path.join(path_speech, 'train'),
                    "val": os.path.join(path_speech, 'val_test'),
                    "test": os.path.join(path_speech, 'val_test')
                   } 
    
    # Check if the files exist
    if os.path.isdir(path_audios) == False:
        raise ValueError(f"Directory {path_audios} does not exist")
    for _, v in dfs_paths.items():
        print(v)
        if v is not None:
            if not os.path.isfile(v):
                raise ValueError(f"File {v} does not exist")
    for d in path_speech.values():
        if not os.path.isdir(d):
            raise ValueError(f"Directory {d} does not exist")
    
    # pass an empty dataframe if the val is None
    dfs = { key: pd.read_csv(val) if val is not None else pd.DataFrame() for key, val in dfs_paths.items() }
    
    ################ Train the model ################
    seed_everything(42) # workers=True
    
    run_name = "SDE_Epochs_{}".format(config["max_epochs"])

    if args.checkpoint is None:
        model = SdeTrainer(sr=config['sampling_frequency'], lr=config["lr"], kernels=kernel, n_grus=n_gru, features_set=features, att_conf=att_conf)
    elif args.checkpoint == 'baseline.ckpt':
        args.checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline.ckpt')
        model = SdeTrainer.load_from_checkpoint(sr=config['sampling_frequency'], lr=config["lr"], kernels=kernel, n_grus=n_gru, features_set=features, att_conf=att_conf, checkpoint_path=args.checkpoint)
    else:
        model = SdeTrainer.load_from_checkpoint(sr=config['sampling_frequency'], lr=config["lr"], kernels=kernel, n_grus=n_gru, features_set=features, att_conf=att_conf, checkpoint_path=args.checkpoint)

    datamodule = SdeDataModule(path_audios, 
                                path_speech, 
                                dfs,
                                sr=config["sampling_frequency"],
                                duration=config["duration"],
                                batch_size = config["batch_size"],
                                dataloader_config=dataloader_config) 
    wandb_logger = WandbLogger(
                    project="SDE-Augmented_RIRs",
                    name=run_name,
                    tags=["augmented_rirs"],
                )
    
    # Save the best model on validation loss, and also the last epoch model
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',  # Metric to monitor
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',  # Checkpoint filename
        save_top_k=1,  # Save only the best model
        mode='min'  # Mode for the monitored metric ('min' or 'max')
    )
    checkpoint_lastep_callback = ModelCheckpoint()
    
    trainer = Trainer(
                        accelerator="gpu",
                        devices = args.gpus,
                        log_every_n_steps = 1,
                        max_epochs=config["max_epochs"],
                        precision = 32,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, checkpoint_lastep_callback],
                        deterministic=args.deterministic
                    )
    wandb_logger.log_hyperparams(config)
    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule)
    
    trainer_test = Trainer(
                        accelerator="gpu",
                        devices = args.gpus,
                        max_epochs=config["max_epochs"],
                        precision = 32,
                        logger=wandb_logger)   
    
    sde_dataset = SdeDataset(path_audios, dfs["test"], path_speech=path_speech["test"], sr=config["sampling_frequency"])  
    sde_dataloader = DataLoader(sde_dataset, batch_size=config["batch_size"])
    trainer_test.test(model, dataloaders=[sde_dataloader])  
    wandb.finish()
    all_results = pd.DataFrame(model.all_test_results)
    all_results.to_csv(run_name + 'test' + ".csv")
