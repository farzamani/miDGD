import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_latent_space(rep, means, samples, labels, color_mapping, epoch, type="Train", save_dir='plot/latent_space.png'):
    # get PCA
    pca = PCA(n_components=2)
    pca.fit(rep)
    rep_pca = pca.transform(rep)
    means_pca = pca.transform(means)
    samples_pca = pca.transform(samples)
    df = pd.DataFrame(rep_pca, columns=["PC1", "PC2"])
    df["type"] = "Representation"
    df["label"] = labels
    df_temp = pd.DataFrame(samples_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM samples"
    df = pd.concat([df,df_temp])
    df_temp = pd.DataFrame(means_pca, columns=["PC1", "PC2"])
    df_temp["type"] = "GMM means"
    df = pd.concat([df,df_temp])

    # make a figure with 2 subplots
    # set a small text size for figures
    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # add spacing between subplots
    fig.subplots_adjust(wspace=0.2)

    # first plot: representations, means and samples
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[3,3,12], alpha=0.8, ax=ax[0], palette=["steelblue","orange","black"])
    ax[0].set_title("E"+str(epoch)+": "+str(type)+" Latent Space (by type)")
    ax[0].legend(loc='upper right', fontsize='small')

    # second plot: representations by label
    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", hue="label", s=3, alpha=0.8, ax=ax[1], palette=color_mapping)
    ax[1].set_title("E"+str(epoch)+": "+str(type)+" Latent Space (by label)")
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3)
    
    plt.savefig(save_dir, dpi=300)
    plt.show()


def plot_sample_recons(dgd, train_loader, sample_index, epoch, type="Train"):
    x_n = [None] * len(sample_index)
    lib_n = [None] * len(sample_index)
    res_n = [None] * len(sample_index)
    
    for i, j in enumerate(sample_index):
        x_n[i], lib_n[i], _ = train_loader.dataset[j]
        x_n[i] = x_n[i].cpu().detach().numpy() + 1
        lib_n[i] = lib_n[i].cpu().detach().numpy()
    
        res_n[i] = dgd.forward(dgd.train_rep.z[j])
        res_n[i] = [tensor.cpu().detach().numpy() for tensor in res_n[i]]
        res_n[i] = (res_n[i] * lib_n[i]) + 1
        res_n[i] = np.array(res_n[i][0])
    
    fig, ax = plt.subplots(ncols=len(sample_index), figsize=(20,4))
    sns.set_theme(style="whitegrid")
    
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_n[i], res_n[i])),
            'type': ['Original'] * len(x_n[i]) + ['Reconstruction'] * len(res_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'{type} sample {j} reconstruction in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)
    
    ax[0].set_ylabel('Frequency')
    plt.show()


def plot_gene_recons(dgd, train_loader, sample_index, epoch):
    x_mrna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i] = train_loader.dataset.data[:,j] + 1
        ## mRNA
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy()
        lib_mrna_n[i] = torch.mean(train_loader.dataset.data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mrna_n[i] = dgd.forward(dgd.train_rep.z)
        ## mRNA
        res_mrna_n[i] = res_mrna_n[i][0].cpu().detach().numpy()
        res_mrna_n[i] = res_mrna_n[i][:,j] 
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_index), figsize=(20,4))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'Gene {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    
    plt.show()


def plot_mirna_recons(dgd, train_loader, sample_index, epoch, type="Train"):
    x_mirna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mirna_n[i] = train_loader.dataset.data[:,j] + 1
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy()
        lib_mirna_n[i] = torch.mean(train_loader.dataset.data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mirna_n[i] = dgd.forward(dgd.train_rep.z)
        ## miRNA
        res_mirna_n[i] = res_mirna_n[i][0].cpu().detach().numpy()
        res_mirna_n[i] = res_mirna_n[i][:,j] 
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_index), figsize=(20,4))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'{type} miRNA {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    
    plt.show()


def plot_mirna_recons_test(dgd, validation_loader, sample_index, epoch, type="Test"):
    x_mirna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    for i, j in enumerate(sample_index):
        # Get training data
        x_mirna_n[i] = validation_loader.dataset.data[:,j] + 1
        ## miRNA
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy()
        lib_mirna_n[i] = torch.mean(validation_loader.dataset.data).cpu().detach().numpy()
        
        # Get gene reconstructions via forward method
        res_mirna_n[i] = dgd.forward(dgd.val_rep.z)
        ## miRNA
        res_mirna_n[i] = res_mirna_n[i][0].cpu().detach().numpy()
        res_mirna_n[i] = res_mirna_n[i][:,j] 
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        
    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_index), figsize=(20,4))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'{type} miRNA {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    
    plt.show()