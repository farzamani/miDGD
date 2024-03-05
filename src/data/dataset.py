import torch
from torch.utils.data import Dataset
import numpy as np

class GeneExpressionDataset(Dataset):
    '''
    Creates a Dataset class for gene expression dataset
    gene_dim is the number of genes (features)
    The rows of the dataframe contain samples, and the
    columns contain gene expression values
    and the class label (tissue) at label_position.
    '''
    def __init__(self, gtex, label_position=-1, scaling_type='mean'):
        '''
        Args:
            gtex: pandas dataframe containing input and output data
            label_position: column id of the class labels
        '''
        self.scaling_type = scaling_type
        self.label_position = label_position

        # convert labels to numbers
        self.label = gtex.iloc[:,label_position].values
        self.data = torch.tensor(gtex.drop(gtex.columns[[self.label_position]], axis=1).values).float()

    def __len__(self):
        return(self.data.shape[0])

    def __getitem__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        expression = self.data[idx,:]
        if self.scaling_type == 'mean':
            lib = torch.mean(expression, dim=-1)
        elif self.scaling_type == 'max':
            lib = torch.max(expression, dim=-1).values
        return expression, lib, idx

    def __getlabel__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        return self.label[idx]