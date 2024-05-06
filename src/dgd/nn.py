import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def logNBdensity(k, m, r):
    """
    Negative Binomial NB(k;m,r), where m is the mean and k is "number of failures"
    r can be real number (and so can k)
    k, and m are tensors of same shape
    r is tensor of shape (1, n_genes)
    Returns the log NB in same shape as k
    """
    # remember that gamma(n+1)=n!
    eps = 1.0e-10  # this is under-/over-flow protection
    x = torch.lgamma(k + r)
    x -= torch.lgamma(r)
    x -= torch.lgamma(k + 1)
    x += k * torch.log(m * (r + m + eps) ** (-1) + eps)
    x += r * torch.log(r * (r + m + eps) ** (-1))
    return x


class OutputModule(nn.Module):
    """
    This is the basis output module class that stands between the decoder and the output data.

    Attributes
    ----------
    fc: torch.nn.modules.container.ModuleList
    n_in: int
        number of hidden units going into this layer
    n_out: int
        number of features that come out of this layer
    distribution: torch.nn.modules.module.Module
        specific class depends on modality argument

    Methods
    ----------
    forward(x)
        input goes through fc modulelist and distribution layer
    loss(model_output,target,scaling_factor,gene_id=None)
        returns loss of distribution for given output
    log_prob(model_output,target,scaling_factor,gene_id=None)
        returns log-prob of distribution for given output
    """

    def __init__(self, fc: torch.nn.modules.container.ModuleList, out_features: int):
        """Args:
        fc: feed-forward NN with at least 1 layer and no last activation function with output dimension
            equal to out_features and input dimension equal to the output dimension of the decoder used
        out_features: number of features from the data modelled by this module
        """
        super(OutputModule, self).__init__()

        self.fc = fc
        self.n_out = out_features

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return x

    """This included DEA but will be discussed later"""


class NB_Module(OutputModule):
    """
    This is the Negative Binomial version of the OutputModule distribution layer.

    Attributes
    ----------
    fc: torch.nn.modules.container.ModuleList
    log_r: torch.nn.parameter.Parameter
        log-dispersion parameter per feature

    Methods
    ----------
    forward(x)
        applies scaling-specific activation
    loss(model_output,target,scaling_factor,gene_id=None)
        returns loss of NB for given output
    log_prob(model_output,target,scaling_factor,gene_id=None)
        returns log-prob of NB for given output
    """

    def __init__(self, fc, out_features, r_init=2, scaling_type="sum"):
        """Args:
        fc: NN (see parent class)
        out_features: number of features from the data modelled by this module
        r_init: initial dispersion factor for all features
        scaling_type: describes type of transformation from model output to targets
            and determines what activation function is used on the output
        """
        super(NB_Module, self).__init__(fc, out_features)

        # substracting 1 now and adding it to the learned dispersion ensures a minimum value of 1
        self.log_r = torch.nn.Parameter(
            torch.full(fill_value=math.log(r_init - 1),
                       size=(1, out_features)),
            requires_grad=True,
        )
        self._scaling_type = scaling_type
        if self._scaling_type == "sum":  # could later re-implement more scalings, but sum is arguably the best so far
            self._activation = "softmax"
        elif self._scaling_type == "mean":
            self._activation = "softplus"
        elif self._scaling_type == "max":
            self._activation = "sigmoid"
        else:
            raise ValueError(
                "scaling_type must be one of 'sum', 'mean', or 'max', but is "
                + self._scaling_type
            )

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        if self._activation == "softmax":
            return F.softmax(x, dim=-1)
        elif self._activation == "softplus":
            return F.softplus(x)
        elif self._activation == "sigmoid":
            return F.sigmoid(x)
        else:
            return x

    @staticmethod
    def rescale(scaling_factor, model_output):
        return scaling_factor * model_output

    def log_prob(self, model_output, target, scaling_factor, feature_id=None):
        # the model output represents the mean normalized count
        # the scaling factor is the used normalization
        if feature_id is not None:  # feature_id could be a single gene
            return logNBdensity(
                target,
                self.rescale(scaling_factor, model_output),
                (torch.exp(self.log_r) + 1)[0, feature_id],
            )
        else:
            return logNBdensity(
                target,
                self.rescale(scaling_factor, model_output),
                (torch.exp(self.log_r) + 1),
            )

    def loss(self, model_output, target, scaling_factor, gene_id=None):
        return -self.log_prob(model_output, target, scaling_factor, gene_id)

    @property
    def dispersion(self):
        return torch.exp(self.log_r) + 1
