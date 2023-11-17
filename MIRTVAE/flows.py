from model import *
from data import *
from helpers import *
import numpy as np
import pandas as pd
import yaml
import random
import torch
from pythae.models import VAE_LinNF, VAE_LinNF_Config, VAE, VAEConfig, BetaVAE, BetaVAEConfig, VAE_IAF, VAE_IAF_Config
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
import math
import os
import time
import sys
from pythae.models import AutoModel

# set working directory to source file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

model_type = sys.argv[1]
iteration = sys.argv[2]
print(f'model_type: {model_type}, iteration: {iteration}')
# set device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']


random.seed(iteration)
theta = np.random.normal(0, 1, cfg['N'] * cfg['mirt_dim']).reshape((cfg['N'], cfg['mirt_dim']))
Q = pd.read_csv(f'parameters/QMatrix{cfg["mirt_dim"]}D.csv', header=None, dtype=float).values

a = np.random.uniform(.5, 2, Q.shape[0] * cfg['mirt_dim']).reshape((Q.shape[0], cfg['mirt_dim']))  # draw discrimination parameters from uniform distribution
a *= Q
b = np.linspace(-2, 2, Q.shape[0], endpoint=True)  # eqally spaced values between -2 and 2 for the difficulty
exponent = np.dot(theta, a.T) + b

prob = np.exp(exponent) / (1 + np.exp(exponent))
data = np.random.binomial(1, prob).astype(float)
# introduce missingness
np.random.seed(cfg['iteration'])
indices = np.random.choice(data.shape[0] * data.shape[1], replace=False,
                           size=int(data.shape[0] * data.shape[1] * cfg['missing_percentage']))
data[np.unravel_index(indices, data.shape)] = float('nan')
data = torch.Tensor(data)

device = "cuda" if torch.cuda.is_available() else "cpu"


class My_Decoder(BaseDecoder):
    def __init__(self, latent_dims, nitems, qm=None):
        BaseDecoder.__init__(self)
        # Initialise network components
        input_layer = latent_dims
        self.activation = nn.Sigmoid()
        self.qm = torch.Tensor(qm).t()

        # create weights and biases
        self.weights = nn.Parameter(torch.randn(input_layer, nitems))  # Manually created weight matrix
        self.bias = nn.Parameter(torch.zeros(nitems))  # Manually created bias vector

        # initialize weights and biases
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        self.weights.data = torch.where(self.qm == 0, torch.zeros_like(self.weights), self.weights)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        pruned_weights = self.weights * self.qm
        out = torch.matmul(x, pruned_weights) + self.bias
        out = self.activation(out)
        output = ModelOutput(
            reconstruction=out  # Set the output from the decoder in a ModelOutput instance
        )
        return output


class My_Encoder(BaseEncoder):
    def __init__(self, latent_dims, nitems, hidden_layer_size, hidden_layer_size2):
        BaseEncoder.__init__(self)
        # initialise netowrk components
        input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size2)
        self.densem = nn.Linear(hidden_layer_size2, latent_dims)
        self.denses = nn.Linear(hidden_layer_size2, latent_dims)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = F.elu(self.dense1(x))
        out = F.elu(self.dense2(out))
        mu = self.densem(out)
        log_sigma = self.denses(out)

        output = ModelOutput(
            embedding=mu,  # Set the output from the decoder in a ModelOutput instance
            log_covariance=2 * log_sigma
        )
        return output

# configure model

if model_type == 'flow':
    model_config = VAE_LinNF_Config(
         input_dim=(1, 28),
         latent_dim=3,
         flows=['Planar', 'Radial', 'Planar'],
         reconstruction_loss='bce',
         activation='elu'
    )

    model = VAE_LinNF(
         model_config=model_config,
         encoder=My_Encoder(3, 28, 20, 10),
         decoder=My_Decoder(3, 28, Q)
    )
elif model_type == 'vae':
    model_config = VAEConfig(
         output_dir='my_model',
         input_dim=(1, 28),
         latent_dim=3,
         reconstruction_loss='bce'
    )

    model = VAE(
         model_config=model_config,
         encoder=My_Encoder(3, 28, 20, 10),
         decoder=My_Decoder(3, 28, Q)
    )
else:
    raise ValueError('Invalid model_type')



config = BaseTrainerConfig(
    output_dir=f'my_model/{model_type}/',
    learning_rate=.005,
    per_device_train_batch_size=10000,
    num_epochs=1500,  # Change this to train the model a bit more
)


pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

start = time.time()
# train model
pipeline(
    train_data=data
)
runtime = time.time() -start

# load trained model
trained_model = model

# compute theta estimates
z = trained_model.encoder(data)[0]
if model_type == 'flow':
    for layer in trained_model.net:
        output = layer(z)
        z = output.out
theta_est = z.detach().numpy()

a_est = trained_model.decoder.weights.t().detach().numpy()
d_est = trained_model.decoder.bias.detach().numpy()

# invert factors
a_est, theta_est = inv_factors(a_est=a_est, theta_est=theta_est, a_true=a)


mse_a = f'{MSE(a_est, a)}\n'
mse_d = f'{MSE(d_est, b)}\n'
mse_theta = f'{MSE(theta_est, theta)}\n'

lll = f'{loglikelihood(a_est, d_est, theta_est, data.numpy())}\n'
runtime = f'{runtime}\n'

# When run with command line arguments, save results to file
with open(f"../results/{'_'.join(sys.argv[1:])}.txt", 'w') as f:
    f.writelines([mse_a, mse_d, mse_theta, lll, runtime])
