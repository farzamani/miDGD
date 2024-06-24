# Import Packages
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb

from base.dgd.nn import NB_Module

from base.utils.helpers import set_seed, get_activation
from base.model.decoder import Decoder
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

# Set random seeds, device and data directory
seed = 0
set_seed(seed)

num_workers = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set results paths
loss_cv_path = 'results/loss_midgd_cv.pickle'
dgd_cv_path = 'results/dgd_midgd_cv.pickle'

# Load data
tcga_mrna_raw = pd.read_table("data/TCGA_mrna_counts_match_iso.tsv", sep='\t', index_col=[0])
tcga_mirna_raw = pd.read_table("data/TCGA_mirna_counts_match_iso.tsv", sep='\t', index_col=[0])

# Filter data
tcga_mrna = data_filtering(tcga_mrna_raw)
tcga_mirna = data_filtering(tcga_mirna_raw)

## Make data split for train and validation sets
mrna_out_dim = tcga_mrna.shape[1]-4
mirna_out_dim = tcga_mirna.shape[1]-4

# shuffle the data
tcga_mrna = tcga_mrna.sample(frac=1, random_state=seed)
tcga_mirna = tcga_mirna.sample(frac=1, random_state=seed)

# Make data split for train, validation, and test sets
train_ratio = 0.70
# Split data
train_mrna, val_mrna = train_test_split(tcga_mrna, train_size=train_ratio, random_state=seed, stratify=tcga_mrna["cancer_type"])
val_mrna, test_mrna = train_test_split(val_mrna, train_size=0.5, random_state=seed, stratify=val_mrna["cancer_type"])

train_idx = train_mrna.index
val_idx = val_mrna.index
combined_idx = train_idx.union(val_idx)

mrna_data = tcga_mrna.loc[combined_idx]
mirna_data = tcga_mirna.loc[combined_idx]

# Model Setup
latent_dim = 20
hidden_dims = [128, 128]
activation = "relu"
reduction_type = "sum" # output loss reduction
scaling_type = "mean"
n_tissues = 32 #len(np.unique(train_mrna['cancer_type']))

# Decoder setup
# set up an output module for the miRNA expression data
mirna_out_fc = nn.Sequential(
    nn.Linear(hidden_dims[-1], 128),
    get_activation(activation),
    nn.Linear(128, mirna_out_dim))
output_mirna_layer = NB_Module(mirna_out_fc, mirna_out_dim, r_init=2, scaling_type=scaling_type)
output_mirna_layer.n_features = mirna_out_dim

# set up an output module for the mRNA expression data
mrna_out_fc = nn.Sequential(
    nn.Linear(hidden_dims[-1], 128),
    get_activation(activation),
    nn.Linear(128, 512),
    get_activation(activation),
    nn.Linear(512, mrna_out_dim))
output_mrna_layer = NB_Module(mrna_out_fc, mrna_out_dim, r_init=2, scaling_type=scaling_type)
output_mrna_layer.n_features = mrna_out_dim

# set up the decoder
decoder = Decoder(latent_dim, 
                    hidden_dims, 
                    output_module_mirna=output_mirna_layer, 
                    output_module_mrna=output_mrna_layer, 
                    activation=activation).to(device)

# Setup gmm init
gmm_mean_scale = 5.0 # usually between 2 and 10
sd_mean_init = 0.2 * gmm_mean_scale / n_tissues # empirically good for single-cell data at dimensionality 20
gmm_spec = {"mean_init": (gmm_mean_scale, 5.0), 
            "sd_init": (sd_mean_init, 1.0), 
            "weight_alpha": 1}

# Setup training parameters
learning_rates={'dec':1e-4,'rep':0.01,'gmm':0.01}
weight_decay=1e-5
betas=(0.5, 0.7)
nepochs = 801
batch_size = 128

pr = 5 # how often to print epoch
plot = 100 # how often to print plot

# Setup sample index for subset
sample_mirna = ['MIMAT0000421', 'MIMAT0000422'] # miR-122-5p and miR-124-3p
sample_index = [tcga_mirna.columns.get_loc(a) for a in sample_mirna]
subset = 1371

is_plot = False
early_stopping = 50

print("Hyperparameters:")
print(f"Latent dimension: {latent_dim}")
print(f"Learning rates: {learning_rates}")
print(f"Weight decay: {weight_decay}")
print(f"betas: {betas}")
print(f"Number of epochs: {nepochs}")
print(f"Hidden dimensions: {hidden_dims}")
print(f"Batch size: {batch_size}")
print(f"Reduction type: {reduction_type}")

# Setup wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="midgd-cv",
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rates,
        "architecture": "midgd",
        "dataset": "midgd",
        "epochs": nepochs,
        "hidden_dim": hidden_dims
    },
    # Set the name of the run  
    name="midgd_cv_run_big",
)

# Run cross-validation training
dgd_cv, loss_cv = cross_validate(mrna_data, mirna_data, batch_size, num_workers, 
                                 decoder, n_tissues, latent_dim, gmm_spec,
                                 learning_rates, weight_decay, betas, reduction_type, scaling_type, nepochs,
                                 pr, plot, sample_index, subset, is_plot=False, n_splits=5, 
                                 mode="midgd", wandb_log=True, seed=seed, early_stopping=early_stopping)

wandb.finish()


# Save model
# Open the file in binary mode
torch.save(loss_cv, loss_cv_path)
torch.save(dgd_cv, dgd_cv_path)

print("The variable 'data' has been saved successfully.")
print(f"Results saved to {loss_cv_path} and {dgd_cv_path}")