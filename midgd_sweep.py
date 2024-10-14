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
from base.data.combined import GeneExpressionDatasetCombined
from base.engine.train import train_midgd
from sklearn.model_selection import train_test_split

# Define sweep config
sweep_configuration = {
    "name": "sweep",
    "method": "random",
    "metric": {"goal": "minimize", "name": "test_loss"},
    "parameters": {
        "activation": {"values": ["leaky_relu","relu"]},
        "latent_dim": {"values": [18, 20, 22, 25, 30]},
        "hidden_dims": {"values": [[100, 100], [100, 128], [100, 100, 100]]},
        "big_mirna": {"values": [True, False]},
        "fc_mirna_1": {"values": [100, 128]},
        "fc_mirna_2": {"values": [128, 256]},
        "fc_mrna_1": {"values": [100, 128]},
        "fc_mrna_2": {"values": [128, 256, 512]},
        "reduction_type": {"value": "sum"},
        "scaling_type": {"values": ["mean"]},
        "n_tissues": {"values": [32, 35, 37, 40]},
        "learning_rates": {
            "parameters": { 
                "dec": {"values": [1e-4]},
                "rep": {"values": [1e-2]},
                "gmm": {"values": [1e-2]}
            }},
        "weight_decay": {
            "parameters": {
                "dec": {"values": [1e-4]},
                "rep": {"values": [1e-4]},
                "gmm": {"values": [1e-4]}
            }},
        "betas": {"values": [(0.5, 0.7), (0.7, 0.9), (0.9, 0.999)]},
        "nepochs": {"value": 1501},
        "early_stopping": {"value": 50},
        "batch_size": {"values": [128, 256]},
        "gmm_mean": {"values": [2.0, 3.0, 4.0, 5.0]},
        "sd_mean": {"values": [0.2, 0.3, 0.4, 0.5]},
        "r_init": {"values": [2, 3, 4]},
        "pi_init": {"values": [0.25, 0.5, 0.75]}

    },
}
def data_filtering(df, filter_zero=True, filter_tumor=True):
    if filter_zero:
        zero_counts = (df == 0).mean()
        selected_features = zero_counts[zero_counts < 0.99].index
        df = df[selected_features]
    if filter_tumor:
        df = df[df['sample_type'].isin(['Primary Tumor', 'Solid Tissue Normal', 'Primary Blood Derived Cancer - Peripheral Blood'])]
    return df

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
    big_mirna = wandb.config.big_mirna
    fc_mirna_1 = wandb.config.fc_mirna_1
    fc_mirna_2 = wandb.config.fc_mirna_2
    fc_mrna_1 = wandb.config.fc_mrna_1
    fc_mrna_2 = wandb.config.fc_mrna_2
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
    early_stopping = wandb.config.early_stopping

    seed = 42
    set_seed(seed)
    num_workers = 14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Number of workers: {num_workers}")
    # Load data
    tcga_mrna_raw = pd.read_table("data/TCGA_mrna_counts_match_iso.tsv", sep='\t', index_col=[0])
    tcga_mirna_raw = pd.read_table("data/TCGA_mirna_counts_match_iso.tsv", sep='\t', index_col=[0])

    # Filter data
    tcga_mrna = data_filtering(tcga_mrna_raw)
    tcga_mirna = data_filtering(tcga_mirna_raw)
    # shuffle the data
    tcga_mrna = tcga_mrna.sample(frac=1, random_state=seed)
    tcga_mirna = tcga_mirna.sample(frac=1, random_state=seed)
    # Make data split for train, validation, and test sets
    train_ratio = 0.7
    # Split data
    train_mrna, val_mrna = train_test_split(tcga_mrna, train_size=train_ratio, random_state=seed, stratify=tcga_mrna["cancer_type"])
    val_mrna, test_mrna = train_test_split(val_mrna, train_size=0.5, random_state=seed, stratify=val_mrna["cancer_type"])

    train_idx = train_mrna.index
    val_idx = val_mrna.index

    train_mirna = tcga_mirna.loc[train_idx]
    val_mirna = tcga_mirna.loc[val_idx]
    
    train_dataset = GeneExpressionDatasetCombined(train_mrna, train_mirna, scaling_type=scaling_type)
    validation_dataset = GeneExpressionDatasetCombined(val_mrna, val_mirna, scaling_type=scaling_type)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False,
                                                    num_workers=num_workers)
    # make data split for train and validation sets
    mrna_out_dim = train_mrna.shape[1]-4
    mirna_out_dim = train_mirna.shape[1]-4

    print(train_loader.dataset.mrna_data.shape)
    print(validation_loader.dataset.mrna_data.shape)

    print(train_loader.dataset.mirna_data.shape)
    print(validation_loader.dataset.mirna_data.shape)

    print(mrna_out_dim)
    print(mirna_out_dim)

    # Decoder setup
    # set up an output module for the miRNA expression data
    if not big_mirna:
        mirna_out_fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], fc_mirna_2),
            get_activation(activation),
            nn.Linear(fc_mirna_2, mirna_out_dim))
        output_mirna_layer = ZINB_Module(mirna_out_fc, mirna_out_dim, r_init=r_init, pi_init=pi_init, scaling_type=scaling_type)
        output_mirna_layer.n_features = mirna_out_dim
    else:
        mirna_out_fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], fc_mirna_1),
            get_activation(activation),
            nn.Linear(fc_mirna_1, fc_mirna_2),
            get_activation(activation),
            nn.Linear(fc_mirna_2, mirna_out_dim))
        output_mirna_layer = ZINB_Module(mirna_out_fc, mirna_out_dim, r_init=r_init, pi_init=pi_init, scaling_type=scaling_type)
        output_mirna_layer.n_features = mirna_out_dim
    
    # set up an output module for the mRNA expression data
    mrna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], fc_mrna_1),
        get_activation(activation),
        nn.Linear(fc_mrna_1, fc_mrna_2),
        get_activation(activation),
        nn.Linear(fc_mrna_2, mrna_out_dim))
    output_mrna_layer = NB_Module(mrna_out_fc, mrna_out_dim, r_init=r_init, scaling_type=scaling_type)
    output_mrna_layer.n_features = mrna_out_dim
    
    # set up the decoder
    decoder = Decoder(latent_dim, 
                      hidden_dims, 
                      output_module_mirna=output_mirna_layer, 
                      output_module_mrna=output_mrna_layer, 
                      activation=activation).to(device)
    
    # setup gmm init
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
    
    sample_mirna = ['MIMAT0000421', 'MIMAT0000422', 'MIMAT0000435', 'MIMAT0000728']
    sample_index = [tcga_mirna.columns.get_loc(a) for a in sample_mirna]
    sample_index
    subset = sample_index[0]

    pr = 5 # how often to print epoch
    plot = 200 # how often to print plot
    
    loss_tab = train_midgd(dgd, train_loader, validation_loader, device,
                           learning_rates=learning_rates, 
                           weight_decay=weight_decay, betas=betas, nepochs=nepochs, fold=None, pr=pr, plot=plot, 
                           reduction_type=reduction_type, scaling_type=scaling_type,
                           sample_index=sample_index, subset=subset, wandb_log=True, early_stopping=early_stopping, is_plot=False)
    
    #torch.save(dgd, f"sweep/midgd/{run.id}_dgd.pickle")
    #torch.save(loss_tab, f"sweep/midgd/{run.id}_loss.pickle")

# Start sweep job.
wandb.agent(sweep_id, project="midgd-sweep", function=main, count=150)