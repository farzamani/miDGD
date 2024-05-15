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
from base.dgd.nn import NB_Module
from base.dgd.DGD import DGD
from base.utils.helpers import set_seed, get_activation
from base.model.decoder_debug import Decoder
from base.engine.cv import cross_validate

from sklearn.model_selection import train_test_split

def data_filtering(df, filter_zero=True, filter_tumor=False):
    if filter_zero:
        zero_counts = (df == 0).mean()
        selected_features = zero_counts[zero_counts < 0.99].index
        df = df[selected_features]
    if filter_tumor:
        df = df[df['sample_type'].isin(['Primary Tumor', 'Solid Tissue Normal'])]
        
    return df

# Define sweep config
sweep_configuration = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "test_r2"},
    "parameters": {
        "fc_mrna": {"values": [128, 256, 512, 1024]},
        "learning_rates": {
            "parameters": { 
                "dec": {"values": [1e-4, 1e-5]},
                "rep": {"value": 0.01},
                "gmm": {"value": 0.01}
            }},
        "weight_decay": {"values": [1e-4, 1e-5, 1e-6]}
    },
}

# Sweep ID
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project="midgd-cv-sweep"
)

def main():
    run = wandb.init()
    # Access hyperparameter values from wandb.config
    fc_mrna = wandb.config.fc_mrna
    learning_rates = wandb.config.learning_rates
    weight_decay = wandb.config.weight_decay

    seed = 42
    set_seed(seed)
    num_workers = 14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    tcga_mrna_raw = pd.read_table("data/TCGA_mrna_counts_match_iso.tsv", sep='\t', index_col=[0])
    tcga_mirna_raw = pd.read_table("data/TCGA_mirna_counts_match_iso.tsv", sep='\t', index_col=[0])
    tcga_mrna = data_filtering(tcga_mrna_raw)
    tcga_mirna = data_filtering(tcga_mirna_raw)

    # Make data split for train, validation, and test sets
    train_ratio= 0.85
    train_mrna, val_mrna = train_test_split(tcga_mrna, train_size=train_ratio, stratify=tcga_mrna['cancer_type'], random_state=seed) 
    # Get the indices of the samples in each split
    train_idx = train_mrna.index
    val_idx = val_mrna.index
    # Use the same indices to split tcga_mirna
    train_mirna = tcga_mirna.loc[train_idx]
    val_mirna = tcga_mirna.loc[val_idx]
    
    # make data split for train and validation sets
    mrna_out_dim = train_mrna.shape[1]-4
    mirna_out_dim = train_mirna.shape[1]-4

    # This one is used for n GMM mixture component 
    n_tissues = len(np.unique(train_mrna['cancer_type']))

    # Hyperparameters for Decoder
    latent_dim = 20 # For the representation layer 
    hidden_dims = [128, 128] # Decoder common hidden dimension
    scaling_type = "mean" # Scaling type for the data
    reduction_type = "sum" # Output loss reduction, you can choose "mean". This is how you want calculate the total loss. 
    activation = "relu" # ["relu", "leaky_relu"]
    r_init = 2

    # Hyperparameters for GMM
    gmm_mean = 5.0 # usually between 2 and 10
    sd_mean = 0.2 # default 0.2

    # Decoder setup
    # set up an output module for the miRNA expression data
    mirna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], 128),
        get_activation(activation),
        nn.Linear(128, mirna_out_dim))
    output_mirna_layer = NB_Module(mirna_out_fc, mirna_out_dim, r_init=r_init, scaling_type=scaling_type)
    output_mirna_layer.n_features = mirna_out_dim
    
    # set up an output module for the mRNA expression data
    mrna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], 128),
        get_activation(activation),
        nn.Linear(128, fc_mrna),
        get_activation(activation),
        nn.Linear(fc_mrna, mrna_out_dim))
    output_mrna_layer = NB_Module(mrna_out_fc, mrna_out_dim, r_init=r_init, scaling_type=scaling_type)
    output_mrna_layer.n_features = mrna_out_dim
    
    # Decoder Setup
    decoder = Decoder(latent_dim, 
                      hidden_dims, 
                      output_module_mirna=output_mirna_layer, 
                      output_module_mrna=output_mrna_layer, 
                      activation=activation).to(device)
    # setup gmm init
    gmm_mean_scale = gmm_mean # usually between 2 and 10
    sd_mean_init = sd_mean * gmm_mean_scale / n_tissues # empirically good for single-cell data at dimensionality 20
    gmm_spec={"mean_init": (gmm_mean_scale, 5.0), "sd_init": (sd_mean_init, 1.0), "weight_alpha": 1}
    
    batch_size = 128
    # Training hyperparameters
    betas = (0.5, 0.7)
    nepochs = 801
    pr = 5 # how often to print epoch
    plot = 200 # how often to print plot
    subset = 1371
    sample_index = [1382, 1310, 34, 360, 
                    765, 999, 2000, 93,
                    0, 10, 20, 300, 
                    123, 345, 456, 567,
                    789, 12, 1050, 56,
                    1371, 2, 1304, 4]
    
    dgd_cv, loss_cv, run = cross_validate(train_mrna, train_mirna, batch_size, device, num_workers, 
                                           decoder, n_tissues, latent_dim, gmm_spec,
                                           learning_rates, weight_decay, betas, reduction_type, nepochs,
                                           pr, plot, sample_index, subset, is_plot=False, n_splits=5, 
                                           mode="all", wandb_log=False, seed=seed, early_stopping=False)
    
    torch.save(dgd_cv, f"models/dgd_cv_{run.id}.pth")
    torch.save(loss_cv, f"metrics/loss_tab_{run.id}.pth")

    # Calculate the average of metrics across all folds
    loss_tab = {metric: [] for metric in loss_cv[0].keys()}

    for metric in loss_tab.keys():
        metric_values = [loss_cv[fold][metric] for fold in loss_cv.keys()]
        loss_tab[metric] = list(np.mean(metric_values, axis=0))
    
    wandb.log(loss_tab)
    
# Start sweep job.
wandb.agent(sweep_id, project="midgd-cv-sweep", function=main, count=10)