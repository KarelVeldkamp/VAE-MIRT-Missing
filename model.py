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
    def __init__(self, file_name):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames
        price_df = pd.read_csv(file_name)
        x = price_df.iloc[:, 1:].values

        self.x_train = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx]


class VariationalEncoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self, latent_dims, hidden_layer_size):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(VariationalEncoder, self).__init__()
        self.dense1 = nn.Linear(28, hidden_layer_size)
        self.dense3m = nn.Linear(hidden_layer_size, latent_dims)
        self.dense3s = nn.Linear(hidden_layer_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :return: a sample from the latent dimensions
        """
        # calculate m and mu based on encoder weights
        x = F.elu(self.dense1(x))
        mu =  self.dense3m(x)
        log_sigma = self.dense3s(x)
        sigma = torch.exp(log_sigma)


        # sample from the latent dimensions
        z = mu + sigma * self.N.sample(mu.shape)

        # calculate kl divergence
        #self.kl = torch.mean(-0.5 * torch.sum(1 + torch.log(sigma) - mu ** 2 - torch.log(sigma).exp(), dim = 1), dim = 0)
        kl = 1 + 2*log_sigma - torch.square(mu) - torch.exp(2*log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * kl

        return z

    def est_theta(self, x):
        """
        estimate theta parameters (latent factor scores)
        :param x: input data
        :return: theta estimates
        """
        x = F.relu(self.dense1(x))
        theta_hat = self.dense3m(x)
        return theta_hat


class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, latent_dims, qm):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        self.linear = nn.Linear(latent_dims, 28)
        self.activation = nn.Sigmoid()

        # remove edges between latent dimensions and items that have a zero in the Q-matrix
        msk_wts = torch.ones((28, 3), dtype=torch.float32)
        for row in range(qm.shape[0]):
            for col in range(qm.shape[1]):
                if qm[row, col] == 0:
                    msk_wts[row][col] = 0

        torch.nn.utils.prune.custom_from_mask(self.linear, name='weight', mask=msk_wts)

    def forward(self, x):
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :return: tensor representing reconstructed item responses
        """
        x = self.linear(x)
        x = self.activation(x)

        return x


class VariationalAutoencoder(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self, latent_dims, hidden_layer_size, qm, learning_rate, batch_size, data_path):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, hidden_layer_size)
        self.decoder = Decoder(latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.data_path = data_path

    def forward(self, x):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :return: tensor representing a reconstruction of the input response data
        """
        z = self.encoder(x)
        return self.decoder(z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass
        X_hat = self(batch)
        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch) * 28
        loss = torch.mean(bce + self.encoder.kl)
        self.log('train_loss',loss)
        return {'loss': loss}

    def train_dataloader(self):
        dataset = ResponseDataset(self.data_path)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader


def train_epoch(vae, dataloader, optimizer):
    """
    Function that trains the model for a single epoch
    :param vae: An instance of VariationalAutoencoder
    :param dataloader: An instance of ResponseDataset
    :param optimizer: A torch optimizer
    :return: This epoch's training loss
    """
    # Set encoder and decoder to train mode
    vae.train()
    train_loss = 0.0
    # Iterate though the data
    for X in dataloader:
        # forward pass
        X_hat = vae(X)
        # loss consistst of reconstruction error and the kl divergence
        #loss = ((X - X_hat)**2).sum() + vae.encoder.kl

        bce = torch.nn.functional.binary_cross_entropy(X_hat, X) * 28
        loss = bce + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader)