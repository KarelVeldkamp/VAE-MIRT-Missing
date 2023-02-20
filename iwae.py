import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model import ResponseDataset


class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int,
                 hidden_layer_size3: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()
        input_layer = nitems*2

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_layer_size2)
        self.dense3 = nn.Linear(hidden_layer_size2, hidden_layer_size3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_layer_size3)
        self.densem = nn.Linear(hidden_layer_size3, latent_dims)
        self.denses = nn.Linear(hidden_layer_size3, latent_dims)


    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """
        # code as tfidf
        #x = x * torch.log(x.shape[0] / torch.sum(x, 0))


        # concatenate the input data with the mask of missing values
        x = torch.cat([x, m], 1)
        # calculate m and mu based on encoder weights
        out = F.elu(self.dense1(x))
        out = self.bn1(out)
        out = F.elu(self.dense2(out))
        out = self.bn2(out)
        out = F.elu(self.dense3(out))
        out = self.bn3(out)
        mu =  self.densem(out)
        log_sigma = self.denses(out)

        return mu, log_sigma


class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int, qm: torch.Tensor=None):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        input_layer = latent_dims
        self.linear = nn.Linear(input_layer, nitems)
        self.activation = nn.Sigmoid()

        # remove edges between latent dimensions and items that have a zero in the Q-matrix
        if qm is not None:
            msk_wts = torch.ones((nitems, input_layer), dtype=torch.float32)
            for row in range(qm.shape[0]):
                for col in range(qm.shape[1]):
                    if qm[row, col] == 0:
                        msk_wts[row][col] = 0
            torch.nn.utils.prune.custom_from_mask(self.linear, name='weight', mask=msk_wts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data is missing
        :return: tensor representing reconstructed item responses
        """
        out = self.linear(x)
        out = self.activation(out)

        return out


class IWAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int,
                 hidden_layer_size3: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 data_path: str,
                 missing: bool=False,
                 beta: int = 1,
                 n_samples=10):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(IWAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.missing = missing

        self.encoder = Encoder(nitems,
                                          latent_dims,
                                          hidden_layer_size,
                                          hidden_layer_size2,
                                          hidden_layer_size3
        )

        self.decoder = Decoder(nitems, latent_dims, qm)
        self.N = torch.distributions.Normal(0, 1)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.data_path = data_path
        self.beta = beta
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """

        mu, log_sigma = self.encoder(x, m)
        sigma = torch.exp(log_sigma)
        # repeat mu and sigma as often as we want to draw samples:
        mu_i = mu.repeat(self.n_samples, 1, 1).permute(1, 0, 2)  # [B x S x I]
        sigma_i = sigma.repeat(self.n_samples, 1, 1).permute(1, 0, 2)  # [B x S x I]
        z = self.sample(mu_i, sigma_i)
        return self.decoder(z), mu, log_sigma

    def sample(self, mu, sigma):
        z = mu + sigma * self.N.sample(mu.shape)
        return z

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data, missing = batch

        mask = (~missing).int()
        reco, mu, log_sigma = self(data, mask)

        # sum the likelihood and the kl divergence
        loss, bce, kl = self.loss(data, reco, mask, mu, log_sigma)

        self.log('binary_cross_entropy', bce)
        self.log('kl_divergence', kl)
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):

        dataset = ResponseDataset(self.data_path)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader

    def loss(self, input, reco, mask, mu, log_sigma):
        input = input.repeat(self.n_samples, 1, 1).permute(1, 0, 2)  # [B x S x I]
        mask = mask.repeat(self.n_samples, 1, 1).permute(1, 0, 2)  # [B x S x I]

        bce = torch.nn.functional.binary_cross_entropy(reco, input, reduction='none') * mask
        bce = torch.mean(bce) * self.nitems
        bce = bce / torch.mean(mask.float())

        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)
        kl = torch.sum(kl, dim=-1)
        kl = -.5 * torch.mean(kl)

        loss = bce+kl

        return loss, bce, kl
