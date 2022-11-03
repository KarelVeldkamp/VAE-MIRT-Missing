import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.utils.prune


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


class VariationalEncoder(nn.Module):
    """
    Neural network used as encoder
    """
    def __init__(self, latent_dims):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(VariationalEncoder, self).__init__()
        self.dense1 = nn.Linear(28, 10)
        self.dense3m = nn.Linear(10, latent_dims)
        self.dense3s = nn.Linear(10, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :return: a sample from the latent dimensions
        """
        # calculate m and mu based on encoder weights
        x = F.relu(self.dense1(x))
        mu =  self.dense3m(x)
        sigma = torch.exp(self.dense3s(x))

        # sample from the latent dimensions
        z = mu + sigma*self.N.sample(mu.shape)

        # calculate kl divergence
        self.kl = torch.mean(-0.5 * torch.sum(1 + torch.log(sigma) - mu ** 2 - torch.log(sigma).exp(), dim = 1), dim = 0)
        return z

    def est_theta(self, x):
        """
        estimate theta parameters (latent factor scores)
        :param x: input data
        :return: theta estimates
        """
        x = F.relu(self.dense1(x))
        theta_hat = self.dense3m(x)
        return(theta_hat)



class Decoder(nn.Module):
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


class VariationalAutoencoder(nn.Module):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self, latent_dims, qm):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims, qm)

    def forward(self, x):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :return: tensor representing a reconstruction of the input response data
        """
        z = self.encoder(x)
        return self.decoder(z)


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
        loss = ((X - X_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)