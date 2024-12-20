import argparse
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import pandas as pd

# from model import SdeTrainer
# from Treble import SdeDataModule
from evaluation.sde_model.model import SdeTrainer
from evaluation.sde_model.dataset import SdeDataModule, SdeDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # add argumense to the script
    parser = argparse.ArgumentParser(description='Train the SDE model on the Treble dataset')
    parser.add_argument('--gpus', type=int, nargs='+', default=[6, 7], help='List of GPUs to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generators')
    parser.add_argument('--deterministic', type=bool, help='Whether to set the random number generators to be deterministic')
    parser.add_argument('--condition', default="condition_2", type=str, help='Condition to test the model on. Options: condition_1 test set are corners , condition_2 test set are centers, condition_3 test set are corners and centers')

    args = parser.parse_args()
    print(args)
    
    ########### Training variables and params ###########
    
    data_dir = '/mnt/data3/SDE'
    path_audios = '/mnt/data3/SDE/raw_IR/treble_GWA'
    path_speech = os.path.join(data_dir, 'VCTK-Corpus' )
    path_dataset_splits = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_splits_treble+gwa')
    
    config = {
    "max_epochs": 60, #50
    "batch_size": 16,
    "lr": 0.001,
    "sampling_frequency": 32000,
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
    
    dfs = { "train": pd.read_csv(os.path.join(path_dataset_splits, "meta_" + args.condition + "_train.csv")),
            "val": pd.read_csv(os.path.join(path_dataset_splits, "meta_" + args.condition + "_val.csv")),
            "test": pd.read_csv(os.path.join(path_dataset_splits, "meta_" + args.condition + "_test.csv")),
    }
    path_speech = { "train": os.path.join(path_speech, 'train'),
                    "val": os.path.join(path_speech, 'val_test'),
                    "test": os.path.join(path_speech, 'val_test'),
                   } 
    
    # Check if the files exist
    if os.path.isdir(path_audios) == False:
        raise ValueError(f"Directory {path_audios} does not exist")
    for w in ["train", "val", "test"]:
        p = os.path.join(path_dataset_splits, "meta_" + args.condition + "_" + w + ".csv")
        if not os.path.isfile(p):
            raise ValueError(f"File {p} does not exist")
    for file in path_speech.values():
        if not os.path.isdir(file):
            raise ValueError(f"File {file} does not exist")
    
    ################ Train the model ################
    seed_everything(42) # workers=True
    
    run_name = "{}_Epochs_{}".format(args.condition, config["max_epochs"])

    model = SdeTrainer(sr=config['sampling_frequency'], lr=config["lr"], kernels=kernel, n_grus=n_gru, features_set=features, att_conf=att_conf)

    datamodule = SdeDataModule(path_audios, 
                                path_speech, 
                                dfs,
                                sr=config["sampling_frequency"],
                                batch_size = config["batch_size"],
                                dataloader_config=dataloader_config) 
    wandb_logger = WandbLogger(
                    project="SDE-Treble-GWA",
                    name=run_name,
                    tags=["TABLE3", "noAirAbsorption"],
                )
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
                        devices = [6],
                        max_epochs=config["max_epochs"],
                        precision = 32,
                        logger=wandb_logger)   
    
    # submission_dir = os.path.join('./evaluation/submission_dataset/condition_1')
    sde_dataset = SdeDataset(path_audios, dfs["test"], path_speech=path_speech["test"], sr=config["sampling_frequency"])  
    sde_dataloader = DataLoader(sde_dataset, batch_size=config["batch_size"])
    trainer_test.test(model, dataloaders=[sde_dataloader])  
    wandb.finish()
    all_results = pd.DataFrame(model.all_test_results)
    all_results.to_csv(run_name + 'test' + ".csv")
