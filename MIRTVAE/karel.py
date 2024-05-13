import torch

from model import *
from data import *
from helpers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import math
import sys
import time
import os
from keras import backend as K

ndim = 3
iteration = sys.argv[1]


##############
# Data generation
###############
N = 1000  # number of subjects
nit = 60  # number of items
ndim = 3  # number of dimensions
nit_dim = int(nit / ndim)  # number of items per dimension
max_nepoch = 50000  # max number of iterations

X = np.full((N, nit), 999)  # empty matrix for item scores
prob = np.full((N, nit), 0.99)  # empty matrix for probability correct
a = np.full((nit, ndim), 0.0)  # empty matrix for discrimination parameters

covMat = np.full((ndim, ndim), 0)  # covariance matrix of dimensions
np.fill_diagonal(covMat, 1)
theta = np.random.multivariate_normal([0] * ndim, covMat, N)  # draw values for the dimensions

structure_matrix = np.full((nit, ndim), 0.0)

for i in range(0, ndim):
    ax = np.random.uniform(.5, 1.5, nit_dim)  # draw discrimination parameters from uniform distribution
    a[range(nit_dim * i, nit_dim * (i + 1)), i] = ax
Q = np.zeros((60,3))  # simple structure configuration
Q[0:20, 0] = 1
Q[20:40, 1] = 1
Q[40:60,2] = 1

b = np.tile(np.linspace(-3, 3, nit_dim, endpoint=True), ndim)  # decoder intercepts

for i in range(0, nit):
    for p in range(0, N):
        prob[p, i] = 1 / (1 + math.exp(-(sum(a[i, :] * theta[p, :]) + b[i])))  # probability correct
        X[p, i] = np.random.binomial(1, prob[p, i])  # draw item scores on basis of prob correct


trainer = Trainer(max_epochs=50000,
                  min_epochs=100,
                  enable_checkpointing=False,
                  logger=None,
                  accelerator='cpu',
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=1e-8, patience=500, mode='min')])

dataset = SimDataset(torch.Tensor(X))
train_loader = DataLoader(dataset, batch_size=1000, shuffle=True)


vae = VAE(nitems=X.shape[1],
           dataloader=train_loader,
           latent_dims=3,
           hidden_layer_size=(nit+ndim)//2,
           qm=Q,
           learning_rate=0.001,
           batch_size=1000,
           n_samples=1,
           sigma_1 = None,
           cor_theta=False)

trainer.fit(vae)

a_est = vae.decoder.weights.t().detach().numpy()
d_est = vae.decoder.bias.t().detach().numpy()

print(((a_est-a)**2).mean())

dataset = SimDataset(torch.Tensor(X))
train_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
data, mask = next(iter(train_loader))
theta_est, log_sigma_est = vae.encoder(data)

mu = theta_est.unsqueeze(0)
sigma = log_sigma_est.unsqueeze(0)
L = torch.tril(vae.transform.weight, -1) + torch.eye(vae.transform.weight.shape[0])


theta_est =  torch.matmul(mu, L).squeeze().detach().numpy()




a_est, theta_est = inv_factors(a_est=a_est, theta_est=theta_est, a_true=a)

print(np.corrcoef(theta_est.T))
cor = np.corrcoef(theta_est.T)

est = np.array([cor[1,0], cor[2,0], cor[1,2]])
true = np.array([covMat[1,0], covMat[2,0], covMat[1,2]])

mse_cor = np.mean((est-true)**2)
mse_a = np.mean((a_est-a)**2)
print(mse_cor)
f = open(f"/Users/karel/Documents/corvae/results/karel_{iteration}.txt", "w+")
f.write(f'{mse_cor}\n')
f.write(f'{mse_a}\n')

# for dim in range(3):
#     plt.figure()
#
#     ai_est = a_est[:, dim]
#     ai_true = a[:, dim]
#
#     plt.scatter(y=ai_est, x=ai_true)
#     plt.plot(ai_true, ai_true)
#     slope, intercept = np.polyfit(ai_true, ai_est, 1)  # Get slope and intercept of the regression line
#     plt.plot(ai_true, slope * np.array(ai_true) + intercept, '--', label='Regression Line',
#              alpha=.3)  # Plot the regression line
#     # for i, x in enumerate(ai_true):
#     #    plt.text(ai_true[i], ai_est[i], i)
#     plt.xlabel('True values')
#     plt.ylabel('Estimates')
#     plt.savefig(f'./karel_a{dim + 1}.png')