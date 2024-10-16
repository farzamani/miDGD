import torch
from torch.utils.data import Dataset
import numpy as np


class GeneExpressionDatasetCombined(Dataset):
    '''
    Creates a Dataset class for gene expression dataset including both mRNA and miRNA data.
    The rows of the dataframe contain samples, and the columns contain gene expression values.
    '''

    def __init__(self, mrna_data, mirna_data, label_position=-1, color_position = -2, 
                 sample_position = -3, tissue_position = -4, scaling_type='mean'):
        '''
        Args:
            mrna_data: pandas dataframe containing mRNA input data
            mirna_data: pandas dataframe containing miRNA input data
            label_position: column id of the class labels (assumed to be the same for both dataframes)
            scaling_type: type of scaling to apply ('mean' or 'max')
        '''
        self.scaling_type = scaling_type
        self.label_position = label_position
        self.color_position = color_position
        self.sample_position = sample_position
        self.tissue_position = tissue_position

        # Assuming labels are the same for both mrna and mirna data and are located in the same column
        self.label = mrna_data.iloc[:, label_position].values
        self.color = mrna_data.iloc[:, color_position].values
        self.sample_type = mrna_data.iloc[:, sample_position].values
        self.tissue_type = mrna_data.iloc[:, tissue_position].values

        # Convert data to tensors and remove label columns
        self.mrna_data = torch.tensor(mrna_data.drop(
            mrna_data.columns[[tissue_position, sample_position, color_position, label_position]], axis=1).values).float()
        self.mirna_data = torch.tensor(mirna_data.drop(
            mirna_data.columns[[tissue_position, sample_position, color_position, label_position]], axis=1).values).float()

    def __len__(self):
        # Assuming both mrna_data and mirna_data have the same number of samples
        return self.mrna_data.shape[0]

    def __getitem__(self, idx):
        if idx is None:
            idx = np.arange(self.__len__())
        mrna_expression = self.mrna_data[idx, :]
        mirna_expression = self.mirna_data[idx, :]

        # Apply scaling if specified
        if self.scaling_type == 'mean':
            mrna_lib = torch.mean(mrna_expression, dim=-1)
            mirna_lib = torch.mean(mirna_expression, dim=-1)
        elif self.scaling_type == 'max':
            mrna_lib = torch.max(mrna_expression, dim=-1).values
            mirna_lib = torch.max(mirna_expression, dim=-1).values
        elif self.scaling_type == 'sum':
            mrna_lib = torch.sum(mrna_expression, dim=-1)
            mirna_lib = torch.sum(mirna_expression, dim=-1)

        return mrna_expression, mirna_expression, mrna_lib, mirna_lib, idx

    def __getlabel__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        return self.label[idx]
