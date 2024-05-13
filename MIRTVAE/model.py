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
        sigma = log_sigma.exp()
        return mu, sigma


class CorrelatedEncoder(pl.LightningModule): # TODO remove class and do within normal encoder
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
        super(CorrelatedEncoder, self).__init__()

        input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.densem = nn.Linear(hidden_layer_size, latent_dims)
        self.denses = nn.Linear(hidden_layer_size, int(latent_dims*(latent_dims+1)/2))


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


        return mu, log_sigma


class SamplingLayer(pl.LightningModule):
    """
    class that samples from the approximate posterior using the reparametrisation trick
    """
    def __init__(self, fixed_sigma=False):
        super(SamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)
        self.fixed_sigma = fixed_sigma

    def forward(self, mu, sigma):
        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)
        if self.fixed_sigma:
            sigma = 1
            #mu = mu/torch.var(mu, 1)

        return mu + sigma * error, sigma, mu + sigma * error

class CorrelatedSamplingLayer(pl.LightningModule):
    """
    class that samples from the approximate posterior using the reparametrisation trick
    """
    def __init__(self, n_samples, latent_dims):
        super(CorrelatedSamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)
        self.n_samples = n_samples

        self.L = torch.nn.Parameter(torch.rand(int(latent_dims * (latent_dims + 1) / 2)))



    def forward(self, mu, sigma):

        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)

        #theta =  mu + sigma * error

        # create empty matrix for L
        L = torch.zeros((mu.shape[1], mu.shape[2], mu.shape[2])).to(mu).repeat(self.n_samples, 1,1,1)
        # fill upped triangular part with encoder output
        triu_indices = torch.triu_indices(mu.shape[2], mu.shape[2], offset=0)
        L[:, :, triu_indices[0], triu_indices[1]] = sigma
        # transpose L to make lowertriangular instead of upper triangular
        L = L.permute(0, 1, 3, 2)

        # make sure cov matrix is positive definite
        R = torch.matrix_exp(L).squeeze()

        sigma = torch.bmm(R, R.permute(0, 2, 1))
        #
        # sds = torch.diag_embed(torch.diagonal(sigma, 0,2))
        #
        # W = torch.bmm(sds, R.permute(0,2,1))
        #
        # theta_hat = torch.bmm(W, theta.squeeze().unsqueeze(-1))
        #potentially move error vector to GPU
        error = error.to(mu)

        # remove importance weighting dimension for now TODO fix
        R = R.squeeze()
        error = error.squeeze()

        # mu + g@e
        sample = mu + torch.bmm(R, error.unsqueeze(-1)).squeeze()



        sigma = torch.bmm(R, R.permute(0,2,1))

        return sample, sigma

class CorrelationTransform(pl.LightningModule):
    def __init__(self, n_samples, latent_dims):
        super(CorrelationTransform, self).__init__()
        self.N = torch.distributions.Normal(0, 1)
        self.n_samples = n_samples
        self.L = torch.nn.Parameter(torch.rand(latent_dims*(latent_dims+1)// 2))

    def forward(self, theta):
        ### Create lower triangular matrix L where lower diagonal elements are trainable and where the first element is fixed to one
        W = self.extract_cholesky_factor(theta.shape[2]).to(theta)

        ### premultiply batch of theta samples by W
        W = W.repeat(theta.shape[1], 1, 1)  # repeat W for number of samples in batch
        theta_hat = torch.bmm(W, theta.squeeze().unsqueeze(-1))  # premultiply each theta in batch with W

        return theta_hat.squeeze().unsqueeze(0)

    def extract_cholesky_factor(self, dim):
        """Extract covariance matrix"""
        L = torch.zeros((dim, dim))  # create empty matrix for L
        triu_indices = torch.triu_indices(dim, dim, offset=0)  # get upper triangular indices
        L[triu_indices[0], triu_indices[1]] = self.L  # fill upper triangular elements of L with trainable parameters
        L[0, 0] = 1
        L = L.t()  # transpose L to make lower triangular instead of upper triangular

        ### Compute cholesky factor of covariance matrix
        R = L# torch.matrix_exp(L)  # matrix exponent to ensure cov matrix is positive definite

        sigma = R @ R.t()
        ### create diagonal matrix with standard deviations on the diagonal
        sds = torch.diag_embed(1 / torch.diagonal(sigma).sqrt())

        ### scale R using the standard deviation matrix
        W = sds @ R

        return W


class LowerTriangularLayer(nn.Module):

    def __init__(self, n):

        super(LowerTriangularLayer, self).__init__()

        self.mask = torch.tril(torch.ones((n,n), dtype=bool))

        self.mask[0,0] = False

        self.L = torch.zeros((n,n))

        self.L[0,0] = 1.0

        self.weight = nn.Parameter(torch.rand(n * (n+1) // 2 - 1))

        self.bias = nn.Parameter(torch.zeros(n))


    def forward(self, input):

        # Apply transformation to weight matrix

        L = self.extract_cholesky_factor()

        L = L.repeat(input.shape[1], 1, 1)  # repeat W for number of samples in batch
        theta_hat = torch.bmm(L, input.squeeze().unsqueeze(-1))  # premultiply each theta in batch with W

        return theta_hat.squeeze().unsqueeze(0)

    def extract_cholesky_factor(self):
        L = self.L.clone()
        L[self.mask] = self.weight
        S = torch.matmul(L, L.t())
        D = torch.diag(1 / torch.diag(S).sqrt())
        L = torch.matmul(D, L)

        return L


class DylanSampling(nn.Module):

    def __init__(self, n):

        super(DylanSampling, self).__init__()

        self.mask = torch.tril(torch.ones((n,n), dtype=bool), diagonal=-1)

        self.L = torch.zeros((n,n))
        self.L = self.L + torch.eye(n)
        self.weight = nn.Parameter(torch.rand(n * (n-1) // 2))
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, sigma):

        # Apply transformation to weight matrix
        error = self.N.sample(mu.shape)

        sample = mu + sigma * error
        L = self.extract_cholesky_factor()


        L = L.repeat(error.shape[1], 1, 1)  # repeat W for number of samples in batch

        sample_transformed = torch.bmm(sample.permute((1, 0, 2)), L).squeeze().unsqueeze(0)  # premultiply each theta in batch with W



        # print(torch.cov(error.squeeze().T))
        # print(torch.cov(mu.squeeze().T))
        #
        # print(torch.cov((mu + sigma * error).squeeze().T))


        return sample, sigma, sample_transformed

    def extract_cholesky_factor(self):
        L = self.L.clone()
        L[self.mask] = self.weight
       # L = (L/torch.norm(L, dim=1, keepdim=True))

        # S = torch.matmul(L, L.t())
        # D = torch.diag(1 / torch.diag(S).sqrt())
        # L = torch.matmul(D, L)

        return L

class CholSampling(nn.Module):

    def __init__(self, n):

        super(CholSampling, self).__init__()

        self.mask = torch.tril(torch.ones((n,n), dtype=bool))

        self.mask[0,0] = False
        self.L = torch.zeros((n,n))
        self.L[0,0] = 1.0
        self.weight = nn.Parameter(torch.rand(n * (n+1) // 2 - 1))
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, sigma): # mu, sigma

        # Apply transformation to weight matrix
        error = self.N.sample(mu.shape)
        sample = mu + sigma * error

        L = self.extract_cholesky_factor()

        # print(L)
        L = L.repeat(error.shape[1], 1, 1)  # repeat W for number of samples in batch
        sample_transformed = torch.bmm(L, sample.squeeze().unsqueeze(-1)).squeeze().unsqueeze(0)  # premultiply each theta in batch with W

        # print(torch.cov(error.squeeze().T))
        # print(torch.cov(mu.squeeze().T))
        #
        # print(torch.cov((mu + sigma * error).squeeze().T))


        return sample, sigma, sample_transformed

    def extract_cholesky_factor(self):
        L = self.L.clone()
        L[self.mask] = self.weight

        # L[1,1] = L[1,1].exp()
        # L[2,2] = L[2,2].exp()
        #
        # L = (L/torch.norm(L, dim=1, keepdim=True))

        S = torch.matmul(L, L.t())
        D = torch.diag(1 / torch.diag(S).sqrt())
        L = torch.matmul(D, L)

        return L


class CholeskyLayer(nn.Module):
    def __init__(self, ndim):
        super(CholeskyLayer, self).__init__()

        self.weight = nn.Parameter(torch.randn((ndim, ndim)))


    def forward(self, theta):
        L = torch.tril(self.weight, -1) + torch.eye(self.weight.shape[0])

        theta_hat =  theta.squeeze() @ L
        return theta_hat.unsqueeze(0)

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
        sigma = F.softplus(log_sigma)

        return mu, sigma


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
                 sigma_1=None,
                 cor_theta:bool=False):
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


        self.cor_theta = cor_theta
        if self.cor_theta:
            #self.transform = CorrelationTransform(n_samples, latent_dims)
            self.transform = LowerTriangularLayer(latent_dims)
        else:
            #self.transform = torch.nn.Identity()
            self.transform = CholeskyLayer(self.latent_dims)


        # of we are estimating correated latent factors
        if sigma_1 is not None:
            dev = next(iter(dataloader))[0].device
            self.cor_post = True # flag variable

            self.mu_1 = torch.zeros(latent_dims).repeat((batch_size, 1)).unsqueeze(-1).to(dev)

            self.sigma_1 = sigma_1.repeat((batch_size, 1, 1)).to(dev)
            self.sigma_1_det = torch.det(sigma_1).repeat((batch_size, 1)).to(dev)
            self.sigma_1_inv = torch.inverse(sigma_1).repeat((batch_size, 1, 1)).to(dev)

            self.sampler = CorrelatedSamplingLayer(n_samples, self.latent_dims)

            self.encoder = CorrelatedEncoder(nitems,
                                             latent_dims,
                                             hidden_layer_size
                                             )
        else:
            self.sampler = SamplingLayer(self.cor_theta)
            #self.sampler = CholSampling(self.latent_dims)
            #self.sampler = DylanSampling(self.latent_dims)
            self.cor_post = False

            self.encoder = Encoder(nitems,
                                     latent_dims,
                                     hidden_layer_size
                                     )






        self.decoder = Decoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.kl = 0
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor):
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
        z, cov, z_transformed = self.sampler(mu, sigma)

        z_transformed = self.transform(z)


        reco = self.decoder(z_transformed)
        return (reco, mu, sigma, z, z_transformed, cov)


    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        return torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': self.lr},
                                 {'params': self.sampler.parameters(), 'lr': self.lr*0.01},
                                 {'params': self.decoder.parameters(), 'lr': self.lr}])


    def training_step(self, batch, batch_idx):
        # if self.encoder.dense1.weight.grad is not None:
        #     print(self.encoder.dense1.weight.grad.mean())
        #     print(self.sampler.weight.grad)

        data, mask = batch
        reco, mu, sigma, z, z_transformed, cov = self(data)


        mask = torch.ones_like(data)
        loss = self.loss(data, reco, mask, mu, cov, z)
        self.log('train_loss',loss)


        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def loss(self, input, reco, mask, mu, sigma, z):
        #calculate log likelihood
        #input = input.unsqueeze(0).repeat(reco.shape[0], 1, 1) # repeat input k times (to match reco size)

        reconstruction_loss = F.binary_cross_entropy(reco.squeeze(), input, reduction='none')
        reconstruction_loss = reconstruction_loss.mean(-1)
        reconstruction_loss *= input.shape[1]

        z_log_sigma = sigma.log().squeeze()
        z_mean = mu.squeeze()

        kl_loss = 1 + 2 * z_log_sigma - torch.square(z_mean) - torch.exp(2 * z_log_sigma)
        kl_loss = torch.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # print(reconstruction_loss.shape)
        # print(kl_loss.shape)
        # exit()

        vae_loss = torch.mean(reconstruction_loss + kl_loss)



        return vae_loss


        log_p_x_theta = ((input * reco).clamp(1e-7).log() + ((1 - input) * (1 - reco)).clamp(1e-7).log()) # compute log ll
        logll = (log_p_x_theta * mask).sum(dim=-1, keepdim=True) # set elements based on missing data to zero

        # # calculate KL divergence
        if self.cor_post:
            sigma = sigma.to(self.sigma_1)
            # squeeze for importance weights, unsqueeze to make column vectors
            mu = mu.squeeze().unsqueeze(-1).to(self.mu_1)


            traces = torch.diagonal(torch.bmm(self.sigma_1_inv, sigma), dim1=1, dim2=2).sum(dim=1)
            # Reshape to get a tensor of size [32, 1]
            traces = traces.view(self.batch_size, 1)

            quad_form = torch.bmm(torch.bmm((self.mu_1-mu).permute(0,2,1), self.sigma_1_inv), (self.mu_1-mu)).squeeze(-1)
            det_frac = torch.log(self.sigma_1_det/torch.det(sigma).unsqueeze(-1))

            kl = (traces + quad_form - self.latent_dims + det_frac)/2


        else:

            log_q_theta_x = torch.distributions.Normal(mu, sigma).log_prob(z).sum(dim=-1, keepdim=True)  # log q(Theta|X)
            log_p_theta = torch.distributions.Normal(torch.zeros_like(z).to(input),
                                                     scale=torch.ones(mu.shape[2]).to(input)).log_prob(z).sum(dim=-1,
                                                                                                              keepdim=True)  # log p(Theta)
            kl = log_q_theta_x - log_p_theta  # kl divergence

        # combine into ELBO
        elbo = logll.to(kl) - kl
        # perform importance weighting
        with torch.no_grad():
            w_tilda = (elbo - elbo.logsumexp(dim=0)).exp()

        loss = (-w_tilda * elbo).sum(0).mean()

        # reconstruction_loss = F.binary_cross_entropy(input, reco.squeeze())
        # reconstruction_loss *= input.shape[1]
        # kl_loss = 1 + 2 * sigma - torch.square(mu) - torch.exp(2 * sigma)
        # kl_loss = torch.sum(kl_loss, -1)
        # kl_loss *= -0.5
        # loss = torch.mean(reconstruction_loss + kl_loss)


        return loss




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
        print(kwargs)
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

        loss = self.loss(data, reco, mask, mu, sigma, z)

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
        mu, sigma = self.encoder(x)
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
        batch = self.data[begin:end, :].clone().detach()
        mask = self.mask[begin:end, :]

        reco, mu, sigma, z = self(batch)

        loss = self.loss(batch, reco, mask, mu, sigma, z)

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
                 mirt_dim: int,
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

        self.encoder = PartialEncoder(self.nitems, emb_dim, h_hidden_dim, latent_dim, hidden_layer_dim, mirt_dim)


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
        loss = self.loss(output, reco, mask, mu, sigma, z)

        # self.log('binary_cross_entropy', bce)
        # self.log('kl_divergence', self.kl)
        self.log('train_loss', loss)
        return {'loss': loss}


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
        loss = self.loss(data, reco, mask, mu, sigma, z)

        self.log('train_loss',loss)

        return {'loss': loss}

