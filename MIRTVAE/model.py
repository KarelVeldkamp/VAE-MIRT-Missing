import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from data import *


class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()

        input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.densem = nn.Linear(hidden_layer_size, latent_dims)
        self.denses = nn.Linear(hidden_layer_size, latent_dims)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights

        out = F.elu(self.dense1(x))
        mu =  self.densem(out)
        log_sigma = self.denses(out)
        #sigma = F.softplus(log_sigma)
        return mu, log_sigma


class SamplingLayer(pl.LightningModule):
    """
    class that samples from the approximate posterior using the reparametrisation trick
    """
    def __init__(self):
        super(SamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, sigma):
        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)
        return mu + sigma.exp() * error


class CholeskyLayer(nn.Module):
    def __init__(self, ndim, n_samples):
        super(CholeskyLayer, self).__init__()

        self.weight = nn.Parameter(torch.eye(ndim))
        self.n_samples = n_samples


    def forward(self, theta):
        L = torch.tril(self.weight, -1) + torch.eye(self.weight.shape[0]).to(self.weight)
        L = L.repeat((self.n_samples, 1,1 ))

        theta_hat =  torch.bmm(theta, L)


        return theta_hat


class ConditionalEncoder(pl.LightningModule):
    """
    Encoder network that takes the mask of missing data as additional input
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(ConditionalEncoder, self).__init__()
        input_layer = nitems*2

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_layer_size)
        self.densem = nn.Linear(hidden_layer_size, latent_dims)
        self.denses = nn.Linear(hidden_layer_size, latent_dims)

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
        #out = F.elu(self.dense2(out))
        #out = self.bn2(out)
        #out = F.elu(self.dense3(out))
        #out = self.bn3(out)
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


        self.dense1 = nn.Linear(latent_dim*5, hidden_layer_dim*2)
        self.dense3m = nn.Linear(hidden_layer_dim*2, mirt_dim)
        self.dense3s = nn.Linear(hidden_layer_dim*2, mirt_dim)

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
        sigma = F.softplus(log_sigma)

        return mu, sigma


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

        input_layer = latent_dims
        self.weights = nn.Parameter(torch.zeros((input_layer, nitems)))  # Manually created weight matrix
        self.bias = nn.Parameter(torch.zeros(nitems))  # Manually created bias vector
        self.activation = nn.Sigmoid()
        if qm is None:
            self.qm = torch.ones((latent_dims, nitems))
        else:
            self.qm = torch.Tensor(qm).t()

    def forward(self, x: torch.Tensor):
        self.qm = self.qm.to(self.weights)
        pruned_weights = self.weights * self.qm
        out = torch.matmul(x, pruned_weights) + self.bias
        out = self.activation(out)

        return out


class VAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 beta: int = 1,
                 n_samples: int = 1,
                 cholesky: bool = False):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.latent_dims = latent_dims
        self.hidden_layer_size = hidden_layer_size

        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims,
                               hidden_layer_size
        )

        self.sampler = SamplingLayer()

        if cholesky:
            self.transform = CholeskyLayer(latent_dims, n_samples)
        else:
            self.transform = nn.Identity()

        self.decoder = Decoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.kl = 0
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, sigma = self.encoder(x)
        # reshape mu and log sigma in order to take multiple samples


        mu = mu.repeat(self.n_samples, 1, 1)
        sigma = sigma.repeat(self.n_samples, 1, 1)
        z = self.sampler(mu, sigma)
        z_tranformed = self.transform(z)
        reco = self.decoder(z_tranformed)
        return reco, mu, sigma, z


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data, mask = batch
        reco, mu, sigma, z = self(data, mask)

        mask = torch.ones_like(data)
        loss, _ = self.loss(data, reco, mask, mu, sigma, z)

        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def loss(self, input, reco, mask, mu, sigma, z):
        #calculate log likelihood
        input = input.unsqueeze(0).repeat(reco.shape[0], 1, 1) # repeat input k times (to match reco size)
        log_p_x_theta = ((input * reco).clamp(1e-7).log() + ((1 - input) * (1 - reco)).clamp(1e-7).log()) # compute log ll
        logll = (log_p_x_theta * mask).sum(dim=-1, keepdim=True) # set elements based on missing data to zero
        #
        # calculate KL divergence
        log_q_theta_x = torch.distributions.Normal(mu, sigma.exp()).log_prob(z).sum(dim = -1, keepdim = True) # log q(Theta|X)
        log_p_theta = torch.distributions.Normal(torch.zeros_like(z).to(input), scale=torch.ones(mu.shape[2]).to(input)).log_prob(z).sum(dim = -1, keepdim = True) # log p(Theta)
        kl =  log_q_theta_x - log_p_theta # kl divergence

        # combine into ELBO
        elbo = logll - kl
        # # perform importance weighting
        with torch.no_grad():
            weight = (elbo - elbo.logsumexp(dim=0)).exp()
        #
        loss = (-weight * elbo).sum(0).mean()


        return loss, weight

    def fscores(self, batch, model, n_mc_samples=50):
        data, mask = batch

        if self.n_samples == 1:
            if model == 'cvae':
                mu, _ = self.encoder(data, mask)
            else:
                mu, _ = self.encoder(data)
            return mu.unsqueeze(0)
        else:
            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims))
            for i in range(n_mc_samples):
                if model == 'cvae':
                    reco, mu, sigma, z = self(data, mask)
                else:
                    reco, mu, sigma, z = self(data)

                loss, weight = self.loss(data, reco, mask, mu, sigma, z)

                idxs = torch.distributions.Categorical(probs=weight.permute(1,2,0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]

                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1, idxs_expanded).squeeze().detach() # Shape [10000, 3]
                scores[i, :, :] = output

            return scores


class CVAE(VAE):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 **kwargs):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(CVAE, self).__init__(**kwargs)
        self.encoder = ConditionalEncoder(self.nitems,
                                          self.latent_dims,
                                          self.hidden_layer_size
        )


    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, sigma = self.encoder(x, m)

        # reshape mu and log sigma in order to take multiple samples
        mu = mu.repeat(self.n_samples, 1, 1)#.permute(1, 0, 2)  # [B x S x I]
        sigma = sigma.repeat(self.n_samples, 1, 1)#.permute(1, 0, 2)  # [B x S x I]

        z = self.sampler(mu, sigma)
        reco = self.decoder(z)

        return reco, mu, sigma, z

    def training_step(self, batch, batch_idx):
        # forward pass
        data, mask = batch
        reco, mu, sigma, z  = self(data, mask)

        loss, _ = self.loss(data, reco, mask, mu, sigma, z)


        self.log('train_loss',loss)

        return {'loss': loss}



class IVAE(VAE):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 data=None,
                 mask=None,
                 **kwargs):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(IVAE, self).__init__(**kwargs)

        self.data = data
        self.mask = mask


        # initialize missing values with (random) reconstruction based on decoder
        with torch.no_grad():
            z =torch.distributions.Normal(0, 1).sample([self.data.shape[0],self.latent_dims])
            gen_data = self.decoder(z)
            self.data[~self.mask.bool()] = gen_data[~self.mask.bool()]



    def forward(self, x: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, sigma = self.encoder(x.to(self.device))
        # repeat mu and sigma as often as we want to draw samples:
        mu = mu.repeat(self.n_samples, 1, 1)
        sigma = sigma.repeat(self.n_samples, 1, 1)
        z = self.sampler(mu, sigma)

        reco = self.decoder(z)



        return reco, mu, sigma, z

    def training_step(self, batch, batch_idx):
        # determine which rows are part of this batch
        begin = batch_idx*self.batch_size
        end = (1+batch_idx)*self.batch_size
        batch = self.data[begin:end, :].clone().detach().to(self.device)
        mask = self.mask[begin:end, :].to(self.device)

        reco, mu, sigma, z = self(batch)

        loss, _ = self.loss(batch, reco, mask, mu, sigma, z)

        self.log('train_loss',loss)

        # update missing data with new probabilities
        with torch.no_grad():
            pred = reco.mean(0)
            copy = batch.clone().detach()
            copy[~mask.bool()] = pred[~mask.bool()]
            self.data[begin:end, :] = batch

        return {'loss': loss}

    def train_dataloader(self):
        dataset = SimDataset(self.data)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader



class PVAE(VAE):
    """
    Neural network for the entire partial variational autoencoder
    """

    def __init__(self,
                 emb_dim: int,
                 h_hidden_dim: int,
                 latent_dim: int,
                 hidden_layer_dim: int,
                 **kwargs):
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
        super(PVAE, self).__init__(**kwargs)
        self.mirt_dim = kwargs['latent_dims']
        self.encoder = PartialEncoder(self.nitems, emb_dim, h_hidden_dim, latent_dim, hidden_layer_dim, self.mirt_dim)


    def forward(self, item_ids, ratings):
        """
        forward pass though the entire network
        :param item_ids: tensor representing user ids
        :param ratings: tensor represeting ratings
        :return: tensor representing a reconstruction of the input response data
        """
        mu, sigma = self.encoder(item_ids, ratings)
        # repeat mu and log sigma in order to take multiple samples
        mu = mu.repeat(self.n_samples, 1, 1)
        sigma = sigma.repeat(self.n_samples, 1, 1)
        z = self.sampler(mu, sigma)
        reco = self.decoder(z)

        return reco, mu, sigma, z

    def training_step(self, batch, batch_idx):
        item_ids, ratings, output, mask = batch
        reco, mu, sigma, z = self(item_ids, ratings)


        # calculate the likelihood, and take the mean of all non missing elements
        loss, _ = self.loss(output, reco, mask, mu, sigma, z)

        # self.log('binary_cross_entropy', bce)
        # self.log('kl_divergence', self.kl)
        self.log('train_loss', loss)
        return {'loss': loss}

    def fscores(self, batch, model='pvae', n_mc_samples=50):
        item_ids, ratings, data, mask = batch

        if self.n_samples == 1:
            mu, _ = self.encoder(item_ids, ratings)
            return mu.unsqueeze(0)
        else:
            scores = torch.empty((n_mc_samples, data.shape[0], self.mirt_dim))

            for i in range(n_mc_samples):
                reco, mu, sigma, z = self(item_ids, ratings)
                loss, weight = self.loss(data, reco, mask, mu, sigma, z)

                idxs = torch.distributions.Categorical(probs=weight.permute(1, 2, 0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]

                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1, idxs_expanded).squeeze().detach()  # Shape [10000, 3]

                scores[i, :, :] = output

            return scores


class IDVAE(VAE):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 **kwargs):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(IDVAE, self).__init__(**kwargs)
        #self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """

        mu, sigma = self.encoder(x)
        # repeat mu and sigma as often as we want to draw samples:
        mu = mu.repeat(self.n_samples, 1, 1)
        sigma = sigma.repeat(self.n_samples, 1, 1)
        z = self.sampler(mu, sigma)
        reco = self.decoder(z)
        return reco, mu, sigma, z

    def training_step(self, batch, batch_idx):
        # forward pass
        data, mask = batch
        reco, mu, sigma, z = self(data)

        # calculate loss
        loss, _ = self.loss(data, reco, mask, mu, sigma, z)

        self.log('train_loss',loss)

        return {'loss': loss}

