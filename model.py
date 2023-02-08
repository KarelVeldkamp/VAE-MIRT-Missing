import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.utils.prune
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class ResponseDataset(Dataset):
    """
    Torch dataset for item response data in csv format
    """
    def __init__(self, file_name: str):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames
        price_df = pd.read_csv(file_name)
        x = price_df.iloc[:, 1:].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.x_missing = torch.isnan(self.x_train)
        self.x_train[torch.isnan(self.x_train)] = 0

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.x_missing[idx]


class VariationalEncoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self, nitems: int, latent_dims: int, hidden_layer_size: int, missing: bool):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(VariationalEncoder, self).__init__()
        self.missing = missing
        if self.missing:
            input_layer = nitems*2
        else:
            input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.dense3m = nn.Linear(hidden_layer_size, latent_dims)
        self.dense3s = nn.Linear(hidden_layer_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """
        # concatenate the input data with the mask of missing values
        if self.missing:
            x = torch.cat([x, m], 1)
        # calculate m and mu based on encoder weights
        out = F.elu(self.dense1(x))
        mu =  self.dense3m(out)
        log_sigma = self.dense3s(out)
        sigma = torch.exp(log_sigma)


        # sample from the latent dimensions
        z = mu + sigma * self.N.sample(mu.shape)

        # calculate kl divergence
        #self.kl = torch.mean(-0.5 * torch.sum(1 + torch.log(sigma) - mu ** 2 - torch.log(sigma).exp(), dim = 1), dim = 0)
        kl = 1 + 2*log_sigma - torch.square(mu) - torch.exp(2*log_sigma)

        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)

        return z

    def est_theta(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        estimate theta parameters (latent factor scores)
        :param x: input data
        :param m: mask representing missing data
        :return: theta estimates
        """
        if self.missing:
            input = torch.cat([x, m], 1)
        else:
            input = x
        x = F.elu(self.dense1(input))
        theta_hat = self.dense3m(x)
        return theta_hat


class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int, qm: torch.Tensor=None, missing: bool=False):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        self.missing = missing
        if self.missing:
            input_layer = nitems+latent_dims
        else:
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

    def forward(self, x: torch.Tensor, m: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data is missing
        :return: tensor representing reconstructed item responses
        """
        if self.missing:
            x = torch.cat([x, m], 1)
        out = self.linear(x)
        out = self.activation(out)

        return out


class VariationalAutoencoder(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 data_path: str,
                 missing: bool=False,
                 beta: int = 1):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VariationalAutoencoder, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.missing = missing

        self.encoder = VariationalEncoder(nitems, latent_dims, hidden_layer_size, missing)

        self.decoder = Decoder(nitems, latent_dims, qm, missing)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.data_path = data_path
        self.beta = beta

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        z = self.encoder(x, m)
        return self.decoder(z, m)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data, missing = batch

        mask = (~missing).int()
        X_hat = self(data, mask)


        # calculate the likelihood, and take the mean of all non missing elements
        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch[0], reduction='none') * mask
        bce = torch.mean(bce)  * self.nitems

        # sum the likelihood and the kl divergence
        loss = bce + self.beta * self.encoder.kl
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):

        dataset = ResponseDataset(self.data_path)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader


