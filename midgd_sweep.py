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

# get the new stuff
from base.utils.helpers import set_seed, get_activation
from base.model.decoder_debug import Decoder
from base.data.combined import GeneExpressionDatasetCombined
from base.engine.train import train_midgd

# Define sweep config
sweep_configuration = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "test_spearman"},
    "parameters": {
        "activation": {"values": ["relu", "leaky_relu"]},
        "latent_dim": {"values": [15, 20, 25]},
        "hidden_dims": {"values": [[128, 128], [128, 256], [256, 256], [128, 256, 256], [128, 128, 256]]},
        "fc_mirna": {"values": [128, 256, 512]},
        "fc_mrna": {"values": [256, 512, 1024]},
        "reduction_type": {"value": "sum"},
        "scaling_type": {"value": "sum"},
        "n_tissues": {"values": [20, 30, 50]},
        "learning_rates": {
            "parameters": { 
                "dec": {"value": 0.001},
                "rep": {"value": 0.01},
                "gmm": {"value": 0.01}
            }},
        "weight_decay": {"value": 0.},
        "betas": {"value": (0.9, 0.999)},
        "nepochs": {"value": 201},
    },
}

os.environ['WANDB_NOTEBOOK_NAME'] = 'tcga_midgd_sweep.ipynb'
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project="midgd-sweep"
)

def main():
    run = wandb.init()

    # Access hyperparameter values from wandb.config
    activation = wandb.config.activation
    latent_dim = wandb.config.latent_dim
    hidden_dims = wandb.config.hidden_dims
    fc_mirna = wandb.config.fc_mirna
    fc_mrna = wandb.config.fc_mrna
    scaling_type = wandb.config.scaling_type
    reduction_type = wandb.config.reduction_type
    n_tissues = wandb.config.n_tissues
    learning_rates = wandb.config.learning_rates
    weight_decay = wandb.config.weight_decay
    betas = wandb.config.betas
    nepochs = wandb.config.nepochs

    seed = 42
    set_seed(seed)
    num_workers = 14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    tcga_mrna = pd.read_table("data/TCGA_mrna_counts_match_iso.tsv", sep='\t', index_col=[0])
    tcga_mirna = pd.read_table("data/TCGA_mirna_counts_match_iso.tsv", sep='\t', index_col=[0])
    # shuffle the data
    tcga_mrna = tcga_mrna.sample(frac=1, random_state=seed)
    tcga_mirna = tcga_mirna.sample(frac=1, random_state=seed)
    # Make data split for train, validation, and test sets
    train_ratio = 0.85
    # Calculate split indices
    total_samples = len(tcga_mrna)
    train_end = int(train_ratio * total_samples)
    # Split the data
    train_mrna = tcga_mrna.iloc[:train_end]
    val_mrna = tcga_mrna.iloc[train_end:]
    train_mirna = tcga_mirna.iloc[:train_end]
    val_mirna = tcga_mirna.iloc[train_end:]

    # Train, val, and test data loaders
    batch_size = 256
    
    # Default scaling_type = "mean"
    train_dataset = GeneExpressionDatasetCombined(train_mrna, train_mirna, scaling_type=scaling_type)
    validation_dataset = GeneExpressionDatasetCombined(val_mrna, val_mirna, scaling_type=scaling_type)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True,
                                                    num_workers=num_workers)
    # make data split for train and validation sets
    mrna_out_dim = train_mrna.shape[1]-4
    mirna_out_dim = train_mirna.shape[1]-4

    # Decoder setup
    # set up an output module for the miRNA expression data
    mirna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], fc_mirna),
        get_activation(activation),
        nn.Linear(fc_mirna, mirna_out_dim))
    output_mirna_layer = NB_Module(mirna_out_fc, mirna_out_dim, scaling_type=scaling_type)
    output_mirna_layer.n_features = mirna_out_dim
    
    # set up an output module for the mRNA expression data
    mrna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], fc_mrna),
        get_activation(activation),
        nn.Linear(fc_mrna, mrna_out_dim))
    output_mrna_layer = NB_Module(mrna_out_fc, mrna_out_dim, scaling_type=scaling_type)
    output_mrna_layer.n_features = mrna_out_dim
    
    # set up the decoder
    decoder = Decoder(latent_dim, hidden_dims, output_mirna_layer, output_mrna_layer, activation=activation).to(device)
    
    # setup gmm init
    gmm_mean_scale = 5.0 # usually between 2 and 10
    sd_mean_init = 0.2 * gmm_mean_scale / n_tissues # empirically good for single-cell data at dimensionality 20
    gmm_spec={"mean_init": (gmm_mean_scale, 5.0), "sd_init": (sd_mean_init, 1.0), "weight_alpha": 1}

    # init a DGD model
    dgd = DGD(
            decoder=decoder,
            n_mix=n_tissues,
            rep_dim=latent_dim,
            gmm_spec=gmm_spec
    )
    
    sample_index = [1382, 1310, 34, 360, 765, 999, 2000, 93, 0, 10, 20, 300, 123, 345, 456, 567, 789, 12, 1050, 56]

    pr = 5 # how often to print epoch
    plot = 50 # how often to print plot
    
    loss_tab = train_midgd(
        dgd, train_loader, validation_loader, device, train_dataset, validation_dataset,
        learning_rates=learning_rates,
        weight_decay=weight_decay, betas=betas, nepochs=nepochs,
        pr=pr, plot=plot, reduction_type=reduction_type,
        sample_index=sample_index, wandb_log=True
    )



# Start sweep job.
wandb.agent(sweep_id, function=main, count=8)