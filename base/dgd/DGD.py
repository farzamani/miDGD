from src.dgd.latent import GaussianMixture
import torch.nn as nn


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
        if len(target) == 1:  # Only one modality is passed
            # mrna_loss = self.decoder.loss(y[0], target[0], scale[0], mod_id=0, reduction=reduction)
            self.dec_loss = self.decoder.loss(
                y[0], target[0], scale[0], mod_id=0, reduction=reduction)
        else:  # Both mRNA and miRNA data are passed
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
