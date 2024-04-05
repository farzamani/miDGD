import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.distributions as D

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from base.utils.helpers import set_seed
from base.utils.helpers import get_activation

from sklearn.decomposition import PCA
from tqdm import tqdm


class GeneExpressionDatasetCombined(Dataset):
    '''
    Creates a Dataset class for gene expression dataset including both mRNA and miRNA data.
    The rows of the dataframe contain samples, and the columns contain gene expression values.
    '''

    def __init__(self, mrna_data, mirna_data, label_position=-1, color_position = -2, scaling_type='mean'):
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

        # Assuming labels are the same for both mrna and mirna data and are located in the same column
        self.label = mrna_data.iloc[:, label_position].values
        self.color = mrna_data.iloc[:, color_position].values

        # Convert data to tensors and remove label columns
        self.mrna_data = torch.tensor(mrna_data.drop(
            mrna_data.columns[[color_position, label_position]], axis=1).values).float()
        self.mirna_data = torch.tensor(mirna_data.drop(
            mirna_data.columns[[color_position, label_position]], axis=1).values).float()

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

        return mrna_expression, mirna_expression, mrna_lib, mirna_lib, idx

    def __getlabel__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        return self.labels[idx]


class DGD(nn.Module):
    def __init__(self, decoder, n_mix, rep_dim, gmm_spec={}):
        super(DGD, self).__init__()
        self.decoder = decoder
        self.rep_dim = rep_dim      # Dimension of representation

        self.gmm = GaussianMixture(n_mix, rep_dim, **gmm_spec)
        self.train_rep = None
        self.val_rep = None
        self.test_rep = None

    def forward(self, z):
        return self.decoder(z)

    def loss(self, z, y, target, scale, gmm_loss=True, reduction="sum"):
        self.dec_loss = self.decoder.loss(
            y, target, scale, reduction=reduction)
        if gmm_loss:
            self.gmm_loss = self.gmm(z)
            if reduction == "mean":
                self.gmm_loss = self.gmm_loss.mean()
            elif reduction == "sum":
                self.gmm_loss = self.gmm_loss.sum()
            return self.dec_loss, self.gmm_loss
        else:
            return self.dec_loss, None

    def forward_and_loss(self, z, target, scale, gmm_loss=True, reduction="sum"):
        y = self.decoder(z)
        return self.loss(z, y, target, scale, gmm_loss, reduction)

    def get_representations(self, type="train"):
        if type == "train":
            return self.train_rep.z.detach().cpu().numpy()
        elif type == "val":
            return self.val_rep.z.detach().cpu().numpy()
        elif type == "test":
            return self.test_rep.z.detach().cpu().numpy()

    def get_gmm_means(self):
        return self.gmm.mean.detach().cpu().numpy()

    def get_latent_space_values(self, rep_type="train", n_samples=1000):
        # get representations
        if rep_type == "train":
            rep = self.train_rep.z.clone().detach().cpu().numpy()
        elif rep_type == "val":
            rep = self.val_rep.z.clone().detach().cpu().numpy()
        elif rep_type == "test":
            rep = self.test_rep.z.clone().detach().cpu().numpy()

        # get gmm means
        gmm_means = self.gmm.mean.clone().detach().cpu().numpy()

        # get some gmm samples
        gmm_samples = self.gmm.sample(n_samples).detach().cpu().numpy()

        return rep, gmm_means, gmm_samples


class RepresentationLayer(torch.nn.Module):
    """
    Implements a representation layer, that accumulates pytorch gradients.

    Representations are vectors in n_rep-dimensional real space. By default
    they will be initialized as a tensor of dimension n_sample x n_rep at origin (zero).

    One can also supply a tensor to initialize the representations (values=tensor).
    The representations will then have the same dimension and will assumes that
    the first dimension is n_sample (and the last is n_rep).

    The representations can be updated once per epoch by standard pytorch optimizers.

    Attributes
    ----------
    n_rep: int
        dimensionality of the representation space
    n_sample: int
        number of samples to be modelled (has to match corresponding dataset)
    z: torch.nn.parameter.Parameter
        tensor of learnable representations of shape (n_sample,n_rep)

    Methods
    ----------
    forward(idx=None)
        takes sample index and returns corresponding representation
    """

    def __init__(self, n_rep: int, n_sample: int, value_init="zero"):
        """Args:
        n_rep: dimensionality of the representation
        n_sample: number of samples to be modelled by the representation
        value_init: per default set to `zero`, leading to an initialization
            of all representations at origin.
            Can also be a tensor of shape (n_sample, n_rep) with custom initialization values
        """
        super(RepresentationLayer, self).__init__()

        self.n_rep = n_rep
        self.n_sample = n_sample

        if value_init == "zero":
            self._value_init = "zero"
            self.z = torch.nn.Parameter(
                torch.zeros(size=(self.n_sample, self.n_rep)), requires_grad=True
            )
        else:
            self._value_init = "custom"
            # Initialize representations from a tensor with values
            assert value_init.shape == (self.n_sample, self.n_rep)
            if isinstance(value_init, torch.Tensor):
                self.z = torch.nn.Parameter(value_init, requires_grad=True)
            else:
                try:
                    self.z = torch.nn.Parameter(
                        torch.Tensor(value_init), requires_grad=True
                    )
                except:
                    raise ValueError(
                        "not able to transform representation init values to torch.Tensor"
                    )

    def forward(self, idx=None):
        """
        Forward pass returns indexed representations
        """
        if idx is None:
            return self.z
        else:
            return self.z[idx]

    def __str__(self):
        return f"""
        RepresentationLayer:
            Dimensionality: {self.n_rep}
            Number of samples: {self.n_sample}
            Value initialization: {self._value_init}
        """


class gaussian:
    """
    This is a simple Gaussian prior used for initializing mixture model means

    Attributes
    ----------
    dim: int
        dimensionality of the space in which samples live
    mean: float
        value of the intended mean of the Normal distribution
    stddev: float
        value of the intended standard deviation

    Methods
    ----------
    sample(n)
        generates samples from the prior
    log_prob(z)
        returns log probability of a vector
    """

    def __init__(self, dim: int, mean: float, stddev: float):
        """Args:
        dim: dimensionality of the latent space
        mean: value for the mean of the prior
        stddev: value for the standard deviation of the prior
        """

        self.dim = dim
        self.mean = mean
        self.stddev = stddev
        self._distrib = torch.distributions.normal.Normal(mean, stddev)

    def sample(self, n):
        """sampling from torch Normal distribution"""
        return self._distrib.sample((n, self.dim))

    def log_prob(self, x):
        """compute log probability of the gaussian prior"""
        return self._distrib.log_prob(x)


class softball:
    """
    Approximate mollified uniform prior.
    It can be imagined as an m-dimensional ball.

    The logistic function creates a soft (differentiable) boundary.
    The prior takes a tensor with a batch of z
    vectors (last dim) and returns a tensor of prior log-probabilities.
    The sample function returns n samples from the prior (approximate
    samples uniform from the m-ball). NOTE: APPROXIMATE SAMPLING.

    Attributes
    ----------
    dim: int
        dimensionality of the space in which samples live
    radius: int
        radius of the m-ball
    sharpness: int
        sharpness of the differentiable boundary

    Methods
    ----------
    sample(n)
        generates samples from the prior with APPROXIMATE sampling
    log_prob(z)
        returns log probability of a vector
    """

    def __init__(self, dim: int, radius: int, sharpness=1):
        """Args:
        dim: dimensionality of the latent space
        radius: radius of the imagined m-ball
        sharpness: sharpness of the differentiable boundary
        """

        self.dim = dim
        self.radius = radius
        self.sharpness = sharpness
        self._norm = math.lgamma(1 + dim * 0.5) - dim * (
            math.log(radius) + 0.5 * math.log(math.pi)
        )

    def sample(self, n):
        """APPROXIMATE sampling of the softball prior"""
        # Return n random samples
        # Approximate: We sample uniformly from n-ball
        with torch.no_grad():
            # Gaussian sample
            sample = torch.randn((n, self.dim))
            # n random directions
            sample.div_(sample.norm(dim=-1, keepdim=True))
            # n random lengths
            local_len = self.radius * \
                torch.pow(torch.rand((n, 1)), 1.0 / self.dim)
            sample.mul_(local_len.expand(-1, self.dim))
        return sample

    def log_prob(self, z):
        """compute log probability of the softball prior"""
        # Return log probabilities of elements of tensor (last dim assumed to be z vectors)
        return self._norm - torch.log(
            1 + torch.exp(self.sharpness * (z.norm(dim=-1) / self.radius - 1))
        )


class GaussianMixture(nn.Module):
    """
    A mixture of multi-variate Gaussians.

    m_mix_comp is the number of components in the mixture
    dim is the dimension of the space
    covariance_type can be "fixed", "isotropic" or "diagonal"
    the mean_prior is initialized as a softball (mollified uniform) with
        mean_init(<radius>, <hardness>)
    log_var_prior is a prior class for the negative log variance of the mixture components
        - log_var = log(sigma^2)
        - If it is not specified, we make this prior a Gaussian from sd_init parameters
        - For the sake of interpretability, the sd_init parameters represent the desired mean and (approximately) sd of the standard deviation
        - the difference btw giving a prior beforehand and giving only init values is that with a given prior, the log_var will be sampled from it, otherwise they will be initialized the same
    alpha determines the Dirichlet prior on mixture coefficients
    Mixture coefficients are initialized uniformly
    Other parameters are sampled from prior

    Attributes
    ----------
    dim: int
        dimensionality of the space
    n_mix_comp: int
        number of mixture components
    mean: torch.nn.parameter.Parameter
        learnable parameter for the GMM means with shape (n_mix_comp,dim)
    log_var: torch.nn.parameter.Parameter
        learnable parameter for the log-variance of the components
        shape depends on what covariances we take into account
            diagonal: (n_mix_comp, dim)
            isotropic: (n_mix_comp)
            fixed: 0
    weight: torch.nn.parameter.Parameter
        learnable parameter for the component weights with shape (n_mix_comp)

    Methods
    ----------
    forward(x)
        returning negative log probability density of a set of representations
    log_prob(x)
        computes summed log probability density
    sample(n_sample)
        creates n_sample random samples from the GMM (taking into account mixture weights)
    component_sample(n_sample)
        creates n_sample new samples PER mixture component
    sample_probs(x)
        computes log-probs per sample (not summed and not including priors)
    sample_new_points(n_points, option='random', n_new=1)
        creating new samples either like in component_sample or from component means
    reshape_targets(y, y_type='true')
        reshaping targets (true counts) to be comparable to model outputs
        from multiple representations per sample
    choose_best_representations(x, losses)
        reducing new representations to 1 per sample (based on lowest loss)
    choose_old_or_new(z_new, loss_new, z_old, loss_old)
        selecting best representation per sample between tow representation tensors (pairwise)
    """

    def __init__(
        self,
        n_mix_comp: int,
        dim: int,
        covariance_type="diagonal",
        mean_init=(2.0, 5.0),
        sd_init=(0.5, 1.0),
        weight_alpha=1,
    ):
        """Args:
        n_mix_comp: number of mixture components
        dim: dimensionality of the latent space (or at least of the corresponding representation)
        covariance_type: string variable determining the portion of the full covariance matrix used
            can be
                `fixed`: all components have the same (not learnable) variance in every dimension
                `isotropic`: every component has 1 learnable variance
                `diagonal`: gives covariance matrix of shape (n_mix_comp,dim)
        mean_init: tuple of mean and std used for the prior over means
            (from which the component means are sampled at initialization)
        sd_init: first value presents the intended mean of the prior over standard deviation
            this prior and corresponding learnable parameter (log_var) are learned as the log-variance
            and the first sd_init value is transformed accordingly.
            The second value presents the standard deviation of the log_var prior.
            This normal distribution over log-variance practically approximates well
            an inverse gamma distribution over variance (found this to be used in Bayesian statistics
             as the marginal posterior for the variance of a Gaussian).
        weight_alpha: concentration parameter of the dirichlet prior
        """
        super().__init__()

        # dimensionality of space and number of components
        self.dim = dim
        self.n_mix_comp = n_mix_comp

        # initialize public parameters
        self._init_means(mean_init)
        self._init_log_var(sd_init, covariance_type)
        self._init_weights(weight_alpha)

        # a dimensionality-dependent term needed in PDF
        self._pi_term = -0.5 * self.dim * math.log(2 * math.pi)

    def _init_means(self, mean_init):
        self._mean_prior = softball(self.dim, mean_init[0], mean_init[1])
        self._mean = nn.Parameter(
            self._mean_prior.sample(self.n_mix_comp), requires_grad=True
        )

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(
        self, value
    ):  # forbid user from changing this parameter outside .load_state_dict
        raise ValueError("GMM mean may not be changed")

    def _init_log_var(self, sd_init, covariance_type):
        # init parameter to learn covariance matrix (as negative log variance to ensure it to be positive definite)
        self._sd_init = sd_init
        self._log_var_factor = self.dim * 0.5  # dimensionality factor in PDF
        self._log_var_dim = 1  # If 'diagonal' the dimension of is dim
        if covariance_type == "fixed":
            # here there are no gradients needed for training
            # this would mainly be used to assume a standard Gaussian
            self._log_var = nn.Parameter(
                torch.empty(self.n_mix_comp, self._log_var_dim), requires_grad=False
            )
        else:
            if covariance_type == "diagonal":
                self._log_var_factor = 0.5
                self._log_var_dim = self.dim
            elif covariance_type != "isotropic":
                raise ValueError(
                    "type must be 'isotropic' (default), 'diagonal', or 'fixed'"
                )

            self._log_var = nn.Parameter(
                torch.empty(self.n_mix_comp, self._log_var_dim), requires_grad=True
            )
        with torch.no_grad():
            self._log_var.fill_(2 * math.log(sd_init[0]))
        self._log_var_prior = gaussian(
            self._log_var_dim, -2 * math.log(sd_init[0]), sd_init[1]
        )
        # this needs to have the negative log variance as a mean to ensure the approximation of
        # an inverse gamma over the variance

    @property
    def log_var(self):
        return self._log_var

    @log_var.setter
    def log_var(
        self, value
    ):  # forbid user from changing this parameter outside .load_state_dict
        raise ValueError("GMM log-variance may not be changed")

    def _init_weights(self, alpha):
        """i.e. Dirichlet prior on mixture"""
        # dirichlet alpha determining the uniformity of the weights
        self._weight_alpha = alpha
        self._dirichlet_constant = math.lgamma(
            self.n_mix_comp * self._weight_alpha
        ) - self.n_mix_comp * math.lgamma(self._weight_alpha)
        # weights are initialized uniformly so that components start out equi-probable
        self._weight = nn.Parameter(torch.ones(
            self.n_mix_comp), requires_grad=True)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(
        self, value
    ):  # forbid user from changing this parameter outside .load_state_dict
        raise ValueError("GMM weigths may not be changed")

    def forward(self, x):
        """
        Forward pass computes the negative log density
        of the probability of z being drawn from the mixture model
        """

        # y = logp = - 0.5k*log(2pi) -(0.5*(x-mean[i])^2)/variance - 0.5k*log(variance)
        # sum terms for each component (sum is over last dimension)
        # x is unsqueezed to (n_sample,1,dim), so broadcasting of mean (n_mix_comp,dim) works
        y = -(x.unsqueeze(-2) - self.mean.to(x.device)
              ).square().div(2 * self.covariance.to(x.device)).sum(-1)
        y = y - self._log_var_factor * self.log_var.to(x.device).sum(-1)
        y = y + self._pi_term

        # For each component multiply by mixture probs
        y = y + torch.log_softmax(self.weight.to(x.device), dim=0)
        y = torch.logsumexp(y, dim=-1)
        y = y + self._prior_log_prob()  # += gives cuda error

        return -y  # returning negative log probability density

    def _prior_log_prob(self):
        """Calculate log prob of prior on mean, log_var, and mixture coefficients"""
        # Mixture weights
        p = self._dirichlet_constant
        if self._weight_alpha != 1:
            p = p + (self._weight_alpha - 1.0) * \
                (self.mixture_probs().log().sum())
        # Means
        p = p + self._mean_prior.log_prob(self.mean).sum()
        # log_var
        if self._log_var_prior is not None:
            p = (
                p + self._log_var_prior.log_prob(-self.log_var).sum()
            )  # ensuring correct approximation
        return p

    def log_prob(self, x):
        """return the log density of the probability of z being drawn from the mixture model"""
        return -self.forward(x)

    def mixture_probs(self):
        """transform weights to mixture probabilites"""
        return torch.softmax(self.weight, dim=-1)

    @property
    def covariance(self):
        """transform negative log variance into covariances"""
        return torch.exp(self.log_var)

    @property
    def stddev(self):
        return torch.sqrt(self.covariance)

    def _Distribution(self):
        """create a distribution from mixture model (for sampling)"""
        with torch.no_grad():
            mix = D.Categorical(probs=torch.softmax(self.weight, dim=-1))
            comp = D.Independent(D.Normal(self.mean, self.stddev), 1)
            return D.MixtureSameFamily(mix, comp)

    def sample(self, n_sample):
        """create samples from the GMM distribution"""
        with torch.no_grad():
            gmm = self._Distribution()
            return gmm.sample(torch.tensor([n_sample]))

    def component_sample(self, n_sample):
        """Returns a sample from each component. Tensor shape (n_sample,n_mix_comp,dim)"""
        with torch.no_grad():
            comp = D.Independent(D.Normal(self.mean, self.stddev), 1)
            return comp.sample(torch.tensor([n_sample]))

    def sample_probs(self, x):
        """compute probability densities per sample without prior. returns tensor of shape (n_sample, n_mix_comp)"""
        y = -(x.unsqueeze(-2) - self.mean).square().div(2 * self.covariance).sum(-1)
        y = y - self._log_var_factor * self.log_var.sum(-1)
        y = y + self._pi_term
        y = y + torch.log_softmax(self.weight, dim=0)
        return torch.exp(y)

    def __str__(self):
        return f"""
        Gaussian_mix_compture:
            Dimensionality: {self.dim}
            Number of components: {self.n_mix_comp}
        """

    def sample_new_points(self, resample_type="mean", n_new_samples=1):
        """
        creates a Tensor with potential new representations.
        These can be drawn from component samples if resample_type is 'sample' or
        from the mean if 'mean'. For drawn samples, n_new_samples defines the number
        of random samples drawn from each component.
        """

        if resample_type == "mean":
            samples = self.mean.clone().cpu().detach()
        else:
            samples = (
                self.component_sample(
                    n_new_samples).view(-1, self.dim).cpu().detach()
            )
        return samples

    def clustering(self, x):
        """compute the cluster assignment (as int) for each sample"""
        return torch.argmax(self.sample_probs(x), dim=-1).to(torch.int16)


class GaussianMixtureSupervised(GaussianMixture):
    """
    Supervised GaussianMixutre class.

    Attributes
    ----------
    Nclass: int
        number of classes to be modeled
    Ncpc: int
        number of components that should model each class
    """

    def __init__(
            self,
            Nclass: int,
            Ncompperclass: int,
            dim: int,
            covariance_type="diagonal",
            mean_init=(2.0, 5.0),
            sd_init=(0.5, 1.0),
            weight_alpha=1
    ):
        super(GaussianMixtureSupervised, self).__init__(
            Nclass*Ncompperclass, dim, covariance_type, mean_init, sd_init, weight_alpha)

        self.Nclass = Nclass  # number of classes in the data
        self.Ncpc = Ncompperclass  # number of components per class

    def forward(self, x, label=None):

        # return unsupervized loss if there are no labels provided
        if label is None:
            y = super().forward(x)
            return y

        y = - (x.unsqueeze(-2).unsqueeze(-2) - self.mean.view(self.Nclass, self.Ncpc, -1)
               ).square().div(2 * self.covariance.view(self.Nclass, self.Ncpc, -1)).sum(-1)
        y = y + self._log_var_factor * \
            self.log_var.view(self.Nclass, self.Ncpc, -1).sum(-1)
        y = y + self._pi_term
        y += torch.log_softmax(self.weight.view(self.Nclass,
                               self.Ncpc), dim=-1)
        y = y.sum(-1)
        # this is replacement for logsumexp of supervised samples
        y = y[(np.arange(y.shape[0]), label)] * self.Nclass

        y = y + self._prior_log_prob()
        return - y

    def label_mixture_probs(self, label):
        return torch.softmax(self.weight[label], dim=-1)

    def supervised_sampling(self, label, sample_type='random'):
        # get samples for each component
        if sample_type == 'origin':
            # choose the component means
            samples = self.mean.clone().detach().unsqueeze(0).repeat(len(label), 1, 1)
        else:
            samples = self.component_sample(len(label))
        # then select the correct component
        return samples[range(len(label)), label]


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


def train_dgd_combined(dgd, train_loader, validation_loader, device, learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01}, 
              weight_decay=0., nepochs=100, pr=1, plot=10, reduction_type="sum", sample_index=[0,11,22,33]):
    if reduction_type == "sum":
        tlen=len(train_loader.dataset)*dgd.decoder.n_out_features
        tlen_gmm=len(train_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
        vlen=len(validation_loader.dataset)*dgd.decoder.n_out_features
        vlen_gmm=len(validation_loader.dataset)*dgd.gmm.n_mix_comp*dgd.gmm.dim
    else:
        tlen=len(train_loader)
        tlen_gmm=len(train_loader)
        vlen=len(validation_loader)
        vlen_gmm=len(validation_loader)
    
    Ntrain=len(train_loader.dataset)
    Nvalidation=len(validation_loader.dataset)
    if dgd.train_rep is None:
        dgd.train_rep = RepresentationLayer(dgd.rep_dim,Ntrain).to(device)

    Nvalidation=len(validation_loader.dataset)
    if dgd.val_rep is None:
        dgd.val_rep = RepresentationLayer(dgd.rep_dim,Nvalidation).to(device)

    dec_optimizer = torch.optim.AdamW(dgd.decoder.parameters(), lr=learning_rates['dec'])
    gmm_optimizer = torch.optim.AdamW(dgd.gmm.parameters(), lr=learning_rates['gmm'])
    train_rep_optimizer = torch.optim.AdamW(dgd.train_rep.parameters(), lr=learning_rates['rep'])
    val_rep_optimizer = torch.optim.AdamW(dgd.val_rep.parameters(), lr=learning_rates['rep'])

    loss_tab = {"Epoch":[],"Train recon":[],"Test recon":[],
                "GMM train":[],"GMM test":[]}
    best_loss=1.e20
    gmm_loss=True

    # For custom color mapping
    color_mapping = dict(zip(train_loader.dataset.label, train_loader.dataset.color))

    # Start training
    for epoch in tqdm(range(nepochs)):
        # Train step
        loss_tab["Epoch"].append(epoch)
        loss_tab["Train recon"].append(0.)
        loss_tab["GMM train"].append(0.)
        train_rep_optimizer.zero_grad()
        dgd.train()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in train_loader:
            dec_optimizer.zero_grad()
            if gmm_loss: gmm_optimizer.zero_grad()
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.train_rep(index),
                target=[mrna_data.to(device), mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device), lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type
            )
            loss_tab["Train recon"][-1] += recon_loss.item()
            loss_tab["GMM train"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
            dec_optimizer.step()
            if gmm_loss: gmm_optimizer.step()
        train_rep_optimizer.step()

        loss_tab["Train recon"][-1] /= tlen
        loss_tab["GMM train"][-1] /= tlen_gmm

        # Validation step
        loss_tab["Test recon"].append(0.)
        loss_tab["GMM test"].append(0.)
        val_rep_optimizer.zero_grad()
        dgd.eval()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in validation_loader:
            recon_loss, gmm_loss = dgd.forward_and_loss(
                z=dgd.val_rep(index),
                target=[mrna_data.to(device), mirna_data.to(device)],  # Pass both mRNA and miRNA data
                scale=[lib_mrna.unsqueeze(1).to(device), lib_mirna.unsqueeze(1).to(device)],  # Pass both scales
                gmm_loss=gmm_loss,
                reduction=reduction_type
            )
            loss_tab["Test recon"][-1] += recon_loss.item()
            loss_tab["GMM test"][-1] += gmm_loss.item()
            loss = recon_loss + gmm_loss
            loss.backward()
        val_rep_optimizer.step()

        loss_tab["Test recon"][-1] /= vlen
        loss_tab["GMM test"][-1] /= vlen_gmm

        if pr>=0 and (epoch)%pr==0:
            print(epoch, loss_tab["Test recon"][-1], loss_tab["GMM test"][-1])
        if plot>=0 and (epoch)%plot==0:
            plot_latent_space(*dgd.get_latent_space_values("train",3000), train_loader.dataset.label, color_mapping, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_sample_recons_combined(dgd, train_loader, sample_index, epoch)
        if plot>=0 and (epoch)%plot==0:
            plot_gene_recons_combined(dgd, train_loader, sample_index, epoch)
    
    return loss_tab


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: int, output_modules: list, activation="relu"):
        super(Decoder, self).__init__()
        # set up the shared decoder
        self.main = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.main.append(nn.Linear(input_dim, hidden_dims[i]))
            self.main.append(get_activation(activation))
            input_dim = hidden_dims[i]

        # set up the modality-specific output module(s)
        self.out_modules = nn.ModuleList()
        for i in range(len(output_modules)):
            self.out_modules.append(output_modules[i])
        self.n_out_groups = len(output_modules)
        self.n_out_features = sum(
            [output_modules[i].n_features for i in range(self.n_out_groups)])

    def forward(self, z):
        for i in range(len(self.main)):
            z = self.main[i](z)
        out = [outmod(z) for outmod in self.out_modules]
        return out

    def log_prob(self, nn_output, target, scale=1, mod_id=None, feature_ids=None, reduction="sum"):
        '''
        Calculating the log probability

        This function is providied with different options of how the output should be provided.
        All log-probs can be summed or averaged into one value (reduction == 'sum' or 'mean'), 
        or not reduced at all (thus giving a log-prob per feature per sample).

        Args:
        nn_output: list of tensors
            the output of the decoder
        target: list of tensors
            the target values
        scale: list of tensors
            the scale of the target values (if applicable, otherwise just 1)
        mod_id: int
            the id of the modality to calculate the log-prob for (if a subset is desired)
        feature_ids: list of int
            the ids of the features to calculate the log-prob for (if a subset is desired, only works if mod_id is not None)
        reduction: str
            the reduction method to use ('sum', 'mean', 'none')
        '''
        if reduction == 'sum':
            log_prob = 0.
            if mod_id is not None:
                log_prob += self.out_modules[mod_id].log_prob(
                    nn_output, target, scale, feature_id=feature_ids).sum()
            else:
                for i in range(self.n_out_groups):
                    log_prob += self.out_modules[i].log_prob(
                        nn_output[i], target[i], scale[i]).sum()
        elif reduction == 'mean':
            log_prob = 0.
            if mod_id is not None:
                log_prob += self.out_modules[mod_id].log_prob(
                    nn_output, target, scale, feature_id=feature_ids).mean()
            else:
                for i in range(self.n_out_groups):
                    log_prob += self.out_modules[i].log_prob(
                        nn_output[i], target[i], scale[i]).mean()
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if mod_id is not None:
                log_prob = self.out_modules[mod_id].log_prob(
                    nn_output, target, scale)
            else:
                n_features = sum(
                    [self.out_modules[i].n_features for i in range(self.n_out_groups)])
                log_prob = torch.zeros(
                    (nn_output[0].shape[0], n_features)).to(dev)
                start_features = 0
                for i in range(self.n_out_groups):
                    log_prob[:, start_features:(start_features+self.out_modules[i].n_features)
                             ] += self.out_modules[i].log_prob(nn_output[i], target[i], scale[i])
                    start_features += self.out_modules[i].n_features
        return log_prob

    def loss(self, nn_output, target, scale=None, mod_id=None, feature_ids=None, reduction="sum"):
        return -self.log_prob(nn_output, target, scale, mod_id, feature_ids, reduction)


def plot_latent_space(rep, means, samples, labels, color_mapping, epoch):
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
    fig.subplots_adjust(wspace=0.4)

    # first plot: representations, means and samples
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", size="type", sizes=[3,3,12], alpha=0.8, ax=ax[0], palette=["steelblue","orange","black"])
    ax[0].set_title("E"+str(epoch)+": Latent space (by type)")
    ax[0].legend(loc='upper right', fontsize='small')

    # second plot: representations by label
    sns.scatterplot(data=df[df["type"] == "Representation"], x="PC1", y="PC2", hue="label", s=3, alpha=0.8, ax=ax[1], palette=color_mapping)
    ax[1].set_title("E"+str(epoch)+": Latent space (by label)")
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, markerscale=3)

    plt.show()


def plot_sample_recons(dgd, train_loader, sample_index, epoch):
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
    
    fig, ax = plt.subplots(ncols=len(sample_index), figsize=(25,4))
    sns.set_theme(style="whitegrid")
    
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_n[i], res_n[i])),
            'type': ['Original'] * len(x_n[i]) + ['Reconstruction'] * len(res_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'Sample {j}reconstruction in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)
    
    ax[0].set_ylabel('Frequency')
    plt.show()


def plot_gene_recons(dgd, train_loader, sample_index, epoch):
    x_n = [None] * len(sample_index)
    lib_n = [None] * len(sample_index)
    res_n = [None] * len(sample_index)
    
    for i, j in enumerate(sample_index):
        x_n[i]  = train_loader.dataset.data[:,j] + 1
        x_n[i] = x_n[i].cpu().detach().numpy()
        
        lib_n[i] = torch.mean(train_loader.dataset.data).cpu().detach().numpy()
        
        res_n[i] = dgd.forward(dgd.train_rep.z) 
        res_n[i] = [tensor.cpu().detach().numpy() for tensor in res_n[i]]
        res_n[i] = res_n[i][0][:,j] 
        res_n[i] = (res_n[i] * lib_n[i]) + 1
    
    fig, ax = plt.subplots(ncols=len(sample_index), figsize=(20,4))
    sns.set_theme(style="whitegrid")
    
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_n[i], res_n[i])),
            'type': ['Original'] * len(x_n[i]) + ['Reconstruction'] * len(res_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'Gene {j} reconstruction in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)
    
    ax[0].set_ylabel('Frequency')
    plt.show()


def plot_sample_recons_combined(dgd, train_loader, sample_index, epoch):
    x_mrna_n = [None] * len(sample_index)
    x_mirna_n = [None] * len(sample_index)
    lib_mrna_n = [None] * len(sample_index)
    lib_mirna_n = [None] * len(sample_index)
    res_mrna_n = [None] * len(sample_index)
    res_mirna_n = [None] * len(sample_index)

    # Get the training data and reconstructions for each sample
    for i, j in enumerate(sample_index):
        # Get training data
        x_mrna_n[i], x_mirna_n[i], lib_mrna_n[i], lib_mirna_n[i], _ = train_loader.dataset[j]
        x_mrna_n[i] = x_mrna_n[i].cpu().detach().numpy() + 1
        lib_mrna_n[i] = lib_mrna_n[i].cpu().detach().numpy()
        x_mirna_n[i] = x_mirna_n[i].cpu().detach().numpy() + 1
        lib_mirna_n[i] = lib_mirna_n[i].cpu().detach().numpy()

        # Get reconstructions via forward method
        res_mrna_n[i], res_mirna_n[i] = dgd.forward(dgd.train_rep.z[j])
        
        res_mrna_n[i] = [tensor.cpu().detach().numpy() for tensor in res_mrna_n[i]]
        res_mrna_n[i] = (res_mrna_n[i] * lib_mrna_n[i]) + 1
        res_mrna_n[i] = np.array(res_mrna_n[i])

        res_mirna_n[i] = [tensor.cpu().detach().numpy() for tensor in res_mirna_n[i]]
        res_mirna_n[i] = (res_mirna_n[i] * lib_mirna_n[i]) + 1
        res_mirna_n[i] = np.array(res_mirna_n[i])

    # Plot initialization and cosmetics
    fig, ax = plt.subplots(nrows=2, ncols=len(sample_index), figsize=(20,8))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style="whitegrid")

    # Plot row 1 for mRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mrna_n[i], res_mrna_n[i])),
            'type': ['Original'] * len(x_mrna_n[i]) + ['Reconstruction'] * len(res_mrna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[0][i], log_scale=True, bins=40)
        ax[0][i].set_title(f'mRNA Sample {j} in epoch {epoch}')
        ax[0][i].set_xlabel('counts+1 (log scale)')
        ax[0][i].set_ylabel(None)

    # Plot row 2 for miRNA
    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_mirna_n[i], res_mirna_n[i])),
            'type': ['Original'] * len(x_mirna_n[i]) + ['Reconstruction'] * len(res_mirna_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[1][i], log_scale=True, bins=40)
        ax[1][i].set_title(f'miRNA Sample {j} in epoch {epoch}')
        ax[1][i].set_xlabel('counts+1 (log scale)')
        ax[1][i].set_ylabel(None)

    ax[0][0].set_ylabel('Frequency')
    ax[1][0].set_ylabel('Frequency')

    plt.show()

def plot_gene_recons_combined(dgd, train_loader, sample_index, epoch):
    x_n = [None] * len(sample_index)
    lib_n = [None] * len(sample_index)
    res_n = [None] * len(sample_index)

    for i, j in enumerate(sample_index):
        x_n[i]  = train_loader.dataset.mrna_data[:,j] + 1
        x_n[i] = x_n[i].cpu().detach().numpy()
        
        lib_n[i] = torch.mean(train_loader.dataset.mrna_data).cpu().detach().numpy()
        
        res_n[i], _ = dgd.forward(dgd.train_rep.z) 
        res_n[i] = res_n[i].cpu().detach().numpy()
        res_n[i] = res_n[i][:,j] 
        res_n[i] = (res_n[i] * lib_n[i]) + 1

    fig, ax = plt.subplots(ncols=len(sample_index), figsize=(20,4))
    sns.set(style="whitegrid")

    for i, j in enumerate(sample_index):
        data = {
            'value': np.concatenate((x_n[i], res_n[i])),
            'type': ['Original'] * len(x_n[i]) + ['Reconstruction'] * len(res_n[i])
        }
        plotdata = pd.DataFrame(data)
        
        sns.histplot(data=plotdata, x='value', hue='type', ax=ax[i], log_scale=True, bins=40)
        ax[i].set_title(f'Gene {j} in epoch {epoch}')
        ax[i].set_xlabel('counts+1 (log scale)')
        ax[i].set_ylabel(None)

    ax[0].set_ylabel('Frequency')
    plt.show()


def main():
    # set random seeds, device and data directory
    seed = 42
    set_seed(seed)

    num_workers = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    datadir = "./data/"

    tcga_mrna_raw = pd.read_table("data/TCGA_mrna_counts_match_iso.tsv", sep='\t', index_col=[0])
    tcga_mirna_raw = pd.read_table("data/TCGA_mirna_counts_match_iso.tsv", sep='\t', index_col=[0])

    # make data split for train and validation sets
    mrna_out_dim = tcga_mrna_raw.shape[1]-2
    mirna_out_dim = tcga_mirna_raw.shape[1]-2

    # shuffle the data
    tcga_mrna_raw = tcga_mrna_raw.sample(frac=1, random_state=seed)
    tcga_mirna_raw = tcga_mirna_raw.sample(frac=1, random_state=seed)

    # make data split for train, validation, and test sets
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    # Calculate split indices
    total_samples = len(tcga_mrna_raw)
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)

    # Split the data
    train_mrna = tcga_mrna_raw.iloc[:train_end]
    val_mrna = tcga_mrna_raw.iloc[train_end:val_end]
    test_mrna = tcga_mrna_raw.iloc[val_end:]

    train_mirna = tcga_mirna_raw.iloc[:train_end]
    val_mirna = tcga_mirna_raw.iloc[train_end:val_end]
    test_mirna = tcga_mirna_raw.iloc[val_end:]

    # Train, val, and test data loaders
    # Default scaling_type = "mean"
    train_dataset = GeneExpressionDatasetCombined(train_mrna, train_mirna)
    validation_dataset = GeneExpressionDatasetCombined(val_mrna, val_mirna)
    test_dataset = GeneExpressionDatasetCombined(test_mrna, test_mirna)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=256, 
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                                    batch_size=256, 
                                                    shuffle=True,
                                                    num_workers=num_workers,
                                                    pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=256, 
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
    

    n_tissues = len(np.unique(train_dataset.label))
    # hyperparameters
    latent_dim = 20
    hidden_dims = [64, 128, 256]
    reduction_type = "sum" # output loss reduction

    # decoder setup

    # set up an output module for the mRNA expression data
    mrna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], mrna_out_dim)
        )
    output_mrna_layer = NB_Module(mrna_out_fc, mrna_out_dim, scaling_type="mean")
    output_mrna_layer.n_features = mrna_out_dim

    # set up an output module for the miRNA expression data
    mirna_out_fc = nn.Sequential(
        nn.Linear(hidden_dims[-1], mirna_out_dim)
        )
    output_mirna_layer = NB_Module(mirna_out_fc, mirna_out_dim, scaling_type="mean")
    output_mirna_layer.n_features = mirna_out_dim


    # set up the decoder
    decoder = Decoder(latent_dim, hidden_dims, [output_mrna_layer, output_mirna_layer]).to(device)

    # init a DGD model

    gmm_mean_scale = 5.0 # usually between 2 and 10
    sd_mean_init = 0.2 * gmm_mean_scale / n_tissues # empirically good for single-cell data at dimensionality 20

    dgd = DGD(
            decoder=decoder,
            n_mix=n_tissues,
            rep_dim=latent_dim,
            gmm_spec={"mean_init": (gmm_mean_scale, 5.0), "sd_init": (sd_mean_init, 1.0), "weight_alpha": 1}
    )

    # train for n epochs and plot learning curves

    n_epochs = 301
    pr = 75 # how often to print epoch
    plot = 75 # how often to print plot

    loss_tab = train_dgd_combined(
        dgd, train_loader, validation_loader, device, 
        learning_rates={'dec':0.001,'rep':0.01,'gmm':0.01},
        weight_decay=0.,nepochs=n_epochs,pr=pr,plot=plot,reduction_type=reduction_type
        )
    
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create a figure to hold the subplots
    plt.figure(figsize=(14, 6))

    # First subplot for Reconstruction loss
    plt.subplot(1, 2, 1)
    sns.lineplot(x="Epoch", y="Train recon", data=loss_tab, label="Train")
    sns.lineplot(x="Epoch", y="Test recon", data=loss_tab, label="Test")
    plt.title("Reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")

    # Second subplot for GMM loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x="Epoch", y="GMM train", data=loss_tab, label="Train")
    sns.lineplot(x="Epoch", y="GMM test", data=loss_tab, label="Test")
    plt.title("GMM loss")
    plt.xlabel("Epoch")
    plt.ylabel("GMM loss")

    # Display the plots
    plt.show()


main()