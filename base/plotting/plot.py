import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_latent_space(rep, means, samples, labels, color_mapping, epoch, fold=0, type="Train", save='latent_space.png'):
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
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # add spacing between subplots
    fig.subplots_adjust(wspace=0.2, top=0.9)

    # first plot: representations, means and samples
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[3,3,12], alpha=0.8, ax=ax[0], palette=["steelblue","orange","black"])
    ax[0].set_title("E"+str(epoch)+": "+str(type)+" Latent Space (by type)")
    ax[0].legend(loc='upper right', fontsize='small')

    # second plot: representations by label
    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", hue="label", s=3, alpha=0.8, ax=ax[1], palette=color_mapping)
    ax[1].set_title("E"+str(epoch)+": "+str(type)+" Latent Space (by label)")
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3)

    plt.suptitle(f'PCA of {type} Latent Space in Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    path = 'plot'
    plt.savefig(os.path.join(path, save+"_"+type+"_F"+str(fold)+"_E"+str(epoch)+".png"))
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

def plot_gene_recons(dgd, loader, sample_index, epoch, device, type='Train', scaling_type="mean"):
    x_mrna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)

    # Get mRNA data
    with torch.no_grad():
        for i, j in enumerate(sample_index):
            # Get training data
            x_mrna_n[i] = loader.dataset.data[:,j].detach().cpu().numpy() + 1

            # Get scaling
            if scaling_type == 'mean':
                scaling = torch.mean(loader.dataset.data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.max(loader.dataset.data, axis=1)
            elif scaling_type == 'sum':
                scaling = torch.sum(loader.dataset.data, axis=1)

            # Get gene reconstructions via forward method
            if type == "Train":
                res_mrna_n[i] = dgd.forward(dgd.train_rep())
            elif type == "Validation":
                res_mrna_n[i] = dgd.forward(dgd.val_rep())
            elif type == "Test":
                res_mrna_n[i] = dgd.forward(dgd.test_rep())
            ## mRNA
            res_mrna_n[i] = res_mrna_n[i] * scaling.unsqueeze(1).to(device)
            res_mrna_n[i] = res_mrna_n[i][:,j].detach().cpu().numpy() + 1
        
    # Create a DataFrame to store the data
    data = []
    for i, j in enumerate(sample_index):
        data.extend([
            {'value': val, 'type': 'Original', 'sample_index': j} for val in x_mrna_n[i]
        ])
        data.extend([
            {'value': val, 'type': 'Reconstruction', 'sample_index': j} for val in res_mrna_n[i]
        ])
    plotdata = pd.DataFrame(data)
    
    # Create the FacetGrid
    sns.set_theme(style="whitegrid")
    
    # Map the histplot to each facet
    g = sns.displot(data=plotdata, x='value', hue='type', bins=40, log_scale=True, col='sample_index', col_wrap=4, height=3, aspect=1.6, legend=True)
    
    # Set the titles and labels
    g.set_titles(col_template=f'{type} gene ' + '{col_name}' + f' in epoch {epoch}')
    g.set_xlabels('counts+1 (log scale)')
    g.set_ylabels('Frequency')
    
    # Adjust the spacing between subplots
    g.tight_layout()
    g.fig.suptitle(f'{type} mRNA Reconstruction in Epoch {epoch}', fontsize=16)
    g.fig.subplots_adjust(top=0.9)  # Adjust the top margin
    
    sns.move_legend(
        g, "upper center",
        bbox_to_anchor=(0.5, -0.01), ncol=2, title=None, frameon=False
    )
    
    # Save the plot
    # path = 'plot'
    # g.savefig(os.path.join(path, save+"_"+type+"_F"+str(fold)+"_E"+str(epoch)+".png"))
    plt.show()


def plot_mirna_recons(dgd, loader, sample_index, epoch, device, type="Train", scaling_type="mean"):
    x_mirna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get mRNA data
    with torch.no_grad():
        for i, j in enumerate(sample_index):
            # Get training data
            x_mirna_n[i] = loader.dataset.data[:,j].detach().cpu().numpy() + 1
            
            # Get scaling
            if scaling_type == 'mean':
                scaling = torch.mean(loader.dataset.data, axis=1)
            elif scaling_type == 'max':
                scaling = torch.max(loader.dataset.data, axis=1)
            elif scaling_type == 'sum':
                scaling = torch.sum(loader.dataset.data, axis=1)

            # Get mirna reconstructions via forward method
            if type == "Train":
                res_mirna_n[i] = dgd.forward(dgd.train_rep())
            elif type == "Validation":
                res_mirna_n[i] = dgd.forward(dgd.val_rep())
            elif type == "Test":
                res_mirna_n[i] = dgd.forward(dgd.test_rep())
            ## miRNA
            res_mirna_n[i] = res_mirna_n[i] * scaling.unsqueeze(1).to(device)
            res_mirna_n[i] = res_mirna_n[i][:,j].detach().cpu().numpy() + 1
        
    # Create a DataFrame to store the data
    data = []
    for i, j in enumerate(sample_index):
        data.extend([
            {'value': val, 'type': 'Original', 'sample_index': j} for val in x_mirna_n[i]
        ])
        data.extend([
            {'value': val, 'type': 'Reconstruction', 'sample_index': j} for val in res_mirna_n[i]
        ])
    plotdata = pd.DataFrame(data)
    
    # Create the FacetGrid
    sns.set_theme(style="whitegrid")
    
    # Map the histplot to each facet
    g = sns.displot(data=plotdata, x='value', hue='type', bins=40, log_scale=True, col='sample_index', col_wrap=4, height=3, aspect=1.6, legend=True)
    
    # Set the titles and labels
    g.set_titles(col_template=f'{type} miRNA ' + '{col_name}' + f' in epoch {epoch}')
    g.set_xlabels('counts+1 (log scale)')
    g.set_ylabels('Frequency')
    
    # Adjust the spacing between subplots
    g.tight_layout()
    g.fig.suptitle(f'{type} miRNA Reconstruction in Epoch {epoch}', fontsize=16)
    g.fig.subplots_adjust(top=0.9)  # Adjust the top margin
    sns.move_legend(
        g, "upper center",
        bbox_to_anchor=(0.5, -0.01), ncol=2, title=None, frameon=False
    )
    
    # Save the plot
    # path = 'plot'
    # g.savefig(os.path.join(path, save+"_"+type+"_F"+str(fold)+"_E"+str(epoch)+".png"))
    plt.show()