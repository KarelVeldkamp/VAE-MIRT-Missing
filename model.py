import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd

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


class SimDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, X, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames


        self.x_train = torch.tensor(X, dtype=torch.float32)
        self.missing = torch.isnan(self.x_train)
        self.x_train[self.missing] = 0
        self.x_train = self.x_train.to(device)
        self.missing = self.x_missing.to(device)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.missing[idx]


class PartialDataset():
    def __init__(self, data, device='cpu'):
        self.data = torch.Tensor(data)
        self.n_max = torch.max(torch.sum(~torch.isnan(self.data), 1))
        self.device = device

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        # output to reconstruct
        output = self.data[idx,:]

        # indicating which ratings are not missing
        mask = torch.Tensor((~torch.isnan(output)).int())

        # save list of movie ids and ratings and replace NA's in output with 0s (will be disregarded in loss using mask)
        item_ids = torch.where(~torch.isnan(output))[0]
        ratings = output[item_ids]
        output = torch.nan_to_num(output,0)

        # convert to tensor and pad with constant values
        item_ids = torch.squeeze(F.pad(item_ids, (self.n_max-ratings.shape[0],0), 'constant', value=0))
        ratings = torch.Tensor(F.pad(ratings, (self.n_max - ratings.shape[0], 0), 'constant', value=2))

        # move tensors to GPU
        item_ids = item_ids.to(self.device)
        ratings = ratings.to(self.device)
        output = output.to(self.device)
        mask = mask.to(self.device)
        return item_ids, ratings, output, mask


class VariationalEncoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(VariationalEncoder, self).__init__()

        input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size2)
        self.densem = nn.Linear(hidden_layer_size2, latent_dims)
        self.denses = nn.Linear(hidden_layer_size2, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights
        out = F.elu(self.dense1(x))
        out = F.elu(self.dense2(out))
        mu =  self.densem(out)
        log_sigma = self.denses(out)
        sigma = torch.exp(log_sigma)


        # sample from the latent dimensions
        z = mu + sigma * self.N.sample(mu.shape)

        # calculate kl divergence
        #self.kl = torch.mean(-0.5 * torch.sum(1 + torch.log(sigma) - mu ** 2 - torch.log(sigma).exp(), dim = 1), dim = 0)
        kl = 1 + 2*log_sigma - torch.square(mu) - torch.exp(2*log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)

        #self.kl =  -0.5 * torch.sum(1 + log_sigma - torch.square(mu) - torch.exp(log_sigma), 1)
        return z, mu, sigma

    def est_theta(self, x: torch.Tensor) -> torch.Tensor:
        """
        estimate theta parameters (latent factor scores)
        :param x: input data
        :param m: mask representing missing data
        :return: theta estimates
        """
        x = F.elu(self.dense1(x))
        theta_hat = self.densem(x)
        return theta_hat


class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()

        input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size2)
        self.densem = nn.Linear(hidden_layer_size2, latent_dims)
        self.denses = nn.Linear(hidden_layer_size2, latent_dims)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights
        out = F.elu(self.dense1(x))
        out = F.elu(self.dense2(out))
        mu =  self.densem(out)
        log_sigma = self.denses(out)

        return mu, log_sigma


class SamplingLayer(pl.LightningModule):
    def __init__(self):
        super(SamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)
        return mu + sigma * error


class ConditionalEncoder(pl.LightningModule):
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
        super(ConditionalEncoder, self).__init__()
        input_layer = nitems*2

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_layer_size2)
        self.dense3 = nn.Linear(hidden_layer_size2, hidden_layer_size3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_layer_size3)
        self.densem = nn.Linear(hidden_layer_size3, latent_dims)
        self.denses = nn.Linear(hidden_layer_size3, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

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


class PartialEncoder(pl.LightningModule):
    def __init__(self, n_items, emb_dim, h_hidden_dim, latent_dim, hidden_layer_dim, mirt_dim):
        """

        :param n_items: total number of items
        :param emb_dim: dimension of the embedding layer
        :param latent_dim: dimension of the latent layer before pooling
        :param hidden_layer_dim: dimension of the hidden layer after pooling
        :param mirt_dim: latent dimension of the distribution that is sampled from
        """
        super(PartialEncoder, self).__init__()
        self.embedding = nn.Embedding(
                n_items+1,
                emb_dim,
        )

        self.emb_dim = emb_dim
        self.h_dense1 = nn.Linear(emb_dim, h_hidden_dim)
        self.h_dense2 = nn.Linear(h_hidden_dim, latent_dim)


        self.dense1 = nn.Linear(latent_dim*5, hidden_layer_dim)
        self.dense3m = nn.Linear(hidden_layer_dim, mirt_dim)
        self.dense3s = nn.Linear(hidden_layer_dim, mirt_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, item_ids: np.array, item_ratings: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param item_ids: a tensor with item ids
        :param item_ratings: a tensor with the corresponding item ratings
        :returns: (sample from the latent distribution, mean of the distribution, sd of the distribution)
        """
        E = self.embedding(item_ids)

        R = item_ratings.unsqueeze(2).repeat((1,1, self.emb_dim))

        S = E * R

        out = F.elu(self.h_dense1(S))
        out = F.elu(self.h_dense2(out))
        mean = torch.mean(out, 1)
        median = torch.quantile(out, .5, 1)
        sd = torch.std(out, 1)
        q25 = torch.quantile(out, .25, 1)
        q75 = torch.quantile(out, .75, 1)
        dist = torch.cat([mean, median, sd, q25, q75], dim=1)
        hidden = F.relu(self.dense1(dist))
        mu = self.dense3m(hidden)
        log_sigma = self.dense3s(hidden)

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


class ConditionalDecoder(pl.LightningModule):
    """
    Neural network used as decoder, conditional on the missing values
    """

    def __init__(self, nitems: int, latent_dims: int, decoder_hidden1:int, decoder_hidden2:int, qm: torch.Tensor=None):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        self.mm1 = nn.Linear(nitems, decoder_hidden1)
        self.mm2 = nn.Linear(decoder_hidden1, decoder_hidden2)

        self.linear = nn.Linear(latent_dims+decoder_hidden2, nitems)

        self.activation = nn.Sigmoid()

        # remove edges between latent dimensions and items that have a zero in the Q-matrix
        if qm is not None:
            msk_wts = torch.ones((nitems, latent_dims+decoder_hidden2), dtype=torch.float32)
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
        m = self.mm1(m.float())
        m = self.mm2(m)
        x = torch.cat([x, m], 1)
        out = self.linear(x)

        out = self.activation(out)

        return out


class CVAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int,
                 hidden_layer_size3: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 beta: int = 1):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(CVAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = ConditionalEncoder(nitems,
                                          latent_dims,
                                          hidden_layer_size,
                                          hidden_layer_size2,
                                          hidden_layer_size3
        )

        self.sampler = SamplingLayer()

        self.decoder = Decoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.kl=0

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, log_sigma = self.encoder(x, m)
        z = self.sampler(mu, log_sigma)
        reco = self.decoder(z)

        # calcualte kl divergence
        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)
        return reco

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data, mask = batch
        X_hat = self(data, mask)

        # calculate the likelihood, and take the mean of all non missing elements
        # bce = data * torch.log(X_hat) + (1 - data) * torch.log(1 - X_hat)
        # bce = bce*mask
        # bce = -self.nitems * torch.mean(bce, 1)
        # bce = bce / torch.mean(mask.float(),1)

        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch[0], reduction='none') * mask
        bce = torch.mean(bce)  * self.nitems
        bce = bce / torch.mean(mask.float())

        # sum the likelihood and the kl divergence

        #loss = torch.mean((bce + self.encoder.kl))
        loss = bce + self.beta * self.kl
        #self.log('binary_cross_entropy', bce)
        #self.log('kl_divergence', self.encoder.kl)
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader


class IVAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2:int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 beta: int = 1,
                 s_miss=None,
                 i_miss=None,
                 data=None):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(IVAE, self).__init__()
        #self.automatic_optimization = False


        self.nitems = nitems

        self.encoder = Encoder(nitems,
                                latent_dims,
                                hidden_layer_size,
                                hidden_layer_size2
        )

        self.sampler = SamplingLayer()

        self.decoder = Decoder(nitems, latent_dims, qm)

        self.data = torch.Tensor(data)

        # initialize missing values with (random) reconstruction based on decoder
        self.s_miss = s_miss
        self.i_miss = i_miss
        with torch.no_grad():
            z =torch.distributions.Normal(0, 1).sample([10000,latent_dims])
            gen_data = self.decoder(z)
            self.data[self.s_miss, self.i_miss] = gen_data[self.s_miss, self.i_miss]

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta


    def forward(self, x: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, log_sigma = self.encoder(x)
        z = self.sampler(mu, log_sigma)
        reco = self.decoder(z)

        # calcualte kl divergence
        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)
        return reco

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        #data = batch
        with torch.no_grad():
            mu, log_sigma = self.encoder(self.data)
            z = self.sampler(mu, log_sigma)
            gen_data = self.decoder(z)

            self.data[self.s_miss, self.i_miss] = gen_data[self.s_miss, self.i_miss]


        X_hat = self(self.data)

        # calculate the likelihood, and take the mean of all non missing elements
        bce = torch.nn.functional.binary_cross_entropy(X_hat, self.data, reduction='none')
        #bce = -self.nitems * torch.mean(self.data * torch.log(X_hat) + (1 - self.data) * torch.log(1 - X_hat), 1)

        bce = torch.mean(bce)  * self.nitems
        #bce = bce / torch.mean(self.mask.float())
        loss = bce + self.beta * self.kl
        # sum the likelihood and the kl divergence
        #loss = torch.mean(bce + self.encoder.kl)
        #self.log('binary_cross_entropy', bce)
        #self.log('kl_divergence', self.encoder.kl)
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        dataset = SimDataset(self.data)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return self.train_loader


class PVAE(pl.LightningModule):
    """
    Neural network for the entire partial variational autoencoder
    """

    def __init__(self,
                 emb_dim: int,
                 h_hidden_dim: int,
                 latent_dim: int,
                 hidden_layer_dim: int,
                 mirt_dim: int,
                 learning_rate: float,
                 batch_size: int,
                 dataloader,
                 Q: np.array,
                 nitems: int,
                 beta: int = 1):
        """

        :param emb_dim: dimension of the item embeddings
        :param latent_dim: dimension of the layer before pooling
        :param hidden_layer_dim: dimension of the layer after pooling
        :param mirt_dim: dimension of the latent distribution to be sampled from
        :param learning_rate: learning rate
        :param batch_size: batch size
        :param dataset: which dataset to use
        """"""
        """
        super(PVAE, self).__init__()
        # self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = PartialEncoder(self.nitems, emb_dim, h_hidden_dim, latent_dim, hidden_layer_dim, mirt_dim)
        self.sampler = SamplingLayer()
        self.decoder = Decoder(self.nitems, mirt_dim, qm=Q)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta

    def forward(self, item_ids, ratings):
        """
        forward pass though the entire network
        :param user_ids: tensor representing user ids
        :param ratings: tensor represeting ratings
        :return: tensor representing a reconstruction of the input response data
        """
        mu, log_sigma = self.encoder(item_ids, ratings)
        z = self.sampler(mu, log_sigma)
        reco = self.decoder(z)

        # calcualte kl divergence
        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)
        return reco

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        item_ids, ratings, output, mask = batch
        X_hat = self(item_ids, ratings)

        # calculate the likelihood, and take the mean of all non missing elements
        bce = torch.nn.functional.binary_cross_entropy(X_hat, output, reduction='none') * mask
        bce = torch.mean(bce) * self.nitems
        bce = bce / torch.mean(mask.float())

        # sum the likelihood and the kl divergence
        loss = bce + self.beta * self.encoder.kl
        self.log('binary_cross_entropy', bce)
        self.log('kl_divergence', self.kl)
        self.log('train_loss', loss)
        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader