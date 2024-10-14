# general imports
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import wandb

# get the DGD modules (once the current version is public we can switch to the repo)
# git clone https://github.com/Center-for-Health-Data-Science/DeepGenerativeDecoder.git
# for now I added them to this repo, but the imports will stay the same
from base.dgd.nn import NB_Module, ZINB_Module
from base.dgd.DGD import DGD

# get the new stuff
from base.utils.helpers import set_seed, get_activation
from base.model.decoder import Decoder
from base.data.dataset import GeneExpressionDataset
from base.engine.train import train_dgd_mirna
from sklearn.model_selection import train_test_split

# Define sweep config
sweep_configuration = {
    "name": "sweep",
    "method": "random",
    "metric": {"goal": "minimize", "name": "test_loss"},
    "parameters": {
        "activation": {"values": ["leaky_relu","relu"]},
        "latent_dim": {"values": [15, 18, 20, 22, 25]},
        "hidden_dims": {"values": [[64, 100], [64, 128], [100, 100], [100, 100, 100], [100, 100, 128], [100, 128], [128, 128], [128, 256], [128, 128, 128], [128, 128, 256], [128, 256, 256], [64, 128, 256], [256, 256, 256]]},
        "reduction_type": {"value": "sum"},
        "scaling_type": {"values": ["mean", "sum"]},
        "n_tissues": {"values": [20, 22, 25, 30, 32, 35, 37, 40]},
        "learning_rates": {
            "parameters": { 
                "dec": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
                "rep": {"values": [1e-1, 1e-2, 1e-3]},
                "gmm": {"values": [1e-1, 1e-2, 1e-3]}
            }},
        "weight_decay": {"values": [1e-3, 1e-4, 1e-5, 1e-6]},
        "betas": {"values": [(0.5, 0.5), (0.5, 0.7), (0.5, 0.9), (0.7, 0.7), (0.7, 0.9), (0.9, 0.999)]},
        "nepochs": {"value": 601},
        "early_stopping": {"value": 50},
        "batch_size": {"values": [64, 128, 256]},
        "gmm_mean": {"values": [2.5, 5.0, 7.5]},
        "sd_mean": {"values": [0.2, 0.4, 0.6, 0.8]},
        "r_init": {"values": [1, 2, 3, 4, 5]},
        "pi_init": {"values": [0.25, 0.5, 0.75]}
    }
}

def data_filtering(df, filter_zero=True, filter_tumor=False):
    if filter_zero:
        zero_counts = (df == 0).mean()
        selected_features = zero_counts[zero_counts < 0.99].index
        df = df[selected_features]
    if filter_tumor:
        df = df[df['sample_type'].isin(['Primary Tumor', 'Solid Tissue Normal'])]
    return df

sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project="mirna-sweep"
)

def main():
    run = wandb.init()

    # Access hyperparameter values from wandb.config
    activation = wandb.config.activation
    latent_dim = wandb.config.latent_dim
    hidden_dims = wandb.config.hidden_dims
    scaling_type = wandb.config.scaling_type
    reduction_type = wandb.config.reduction_type
    n_tissues = wandb.config.n_tissues
    learning_rates = wandb.config.learning_rates
    weight_decay = wandb.config.weight_decay
    betas = wandb.config.betas
    nepochs = wandb.config.nepochs
    batch_size = wandb.config.batch_size
    gmm_mean = wandb.config.gmm_mean
    sd_mean = wandb.config.sd_mean
    r_init = wandb.config.r_init
    pi_init = wandb.config.pi_init

    seed = 42
    set_seed(seed)
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Number of workers: {num_workers}")
    # Load data
    tcga_mirna_raw = pd.read_table("data/TCGA_mirna_counts_match_iso.tsv", sep='\t', index_col=[0])

    # Filter data
    tcga_mirna = data_filtering(tcga_mirna_raw)
    # shuffle the data
    tcga_mirna = tcga_mirna.sample(frac=1, random_state=seed)
    # Make data split for train, validation, and test sets
    train_ratio = 0.7
    # Split data
    train_mirna, val_mirna = train_test_split(tcga_mirna, train_size=train_ratio, stratify=tcga_mirna['cancer_type'], random_state=seed) 
    val_mirna, test_mirna = train_test_split(val_mirna, test_size=0.50, stratify=val_mirna['cancer_type'], random_state=seed)

    # Train, val, and test data loaders
    # Default scaling_type = "mean"
    train_dataset = GeneExpressionDataset(train_mirna, scaling_type=scaling_type)
    validation_dataset = GeneExpressionDataset(val_mirna, scaling_type=scaling_type)
    test_dataset = GeneExpressionDataset(test_mirna, scaling_type=scaling_type)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False,
                                                    num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False,
                                            num_workers=num_workers)
    
    # make data split for train and validation sets
    mirna_out_dim = train_mirna.shape[1]-4

    print(train_mirna.shape)
    print(val_mirna.shape)
    print(test_mirna.shape)
    print(mirna_out_dim)

    # Decoder setup
    # Output Module Setup
    mirna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], mirna_out_dim)
        )
    output_mirna_layer = NB_Module(mirna_out_fc, mirna_out_dim, r_init=r_init, scaling_type=scaling_type)
    output_mirna_layer.n_features = mirna_out_dim

    # Set up the decoder
    decoder = Decoder(latent_dim, hidden_dims, output_module=output_mirna_layer).to(device)

    # Setup GMM init
    gmm_mean_scale = gmm_mean # usually between 2 and 10
    sd_mean_init = sd_mean * gmm_mean_scale / n_tissues # empirically good for single-cell data at dimensionality 20
    gmm_spec={"mean_init": (gmm_mean_scale, 5.0), "sd_init": (sd_mean_init, 1.0), "weight_alpha": 1}

    # init a DGD model
    dgd = DGD(
            decoder=decoder,
            n_mix=n_tissues,
            rep_dim=latent_dim,
            gmm_spec=gmm_spec
    )
    
    print(dgd)

    sample_mirna = ['MIMAT0000421', 'MIMAT0000422', 'MIMAT0000435', 'MIMAT0000728']
    sample_index = [tcga_mirna.columns.get_loc(a) for a in sample_mirna]
    sample_index
    subset = sample_index[0]

    pr = 5 # how often to print epoch
    plot = 200 # how often to print plot
    
    # Train the model
    loss_tab = train_dgd_mirna(
        dgd, train_loader, validation_loader, device,
        learning_rates=learning_rates, weight_decay=weight_decay, betas=betas, 
        nepochs=nepochs, pr=pr, plot=plot, reduction_type=reduction_type, 
        sample_index=sample_index, subset=subset, wandb_log=True, early_stopping=50, is_plot=False)
    
    torch.save(dgd, f"sweep/mirna/{run.id}_model_mirna.pickle")
    torch.save(loss_tab, f"sweep/mirna/{run.id}_loss_mirna.pickle")

# Start sweep job.
wandb.agent(sweep_id, project="mirna-sweep", function=main, count=150)