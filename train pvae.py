from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger


def inv_factors(a, theta=None):
    """
    Helper function that inverts factors when discrimination values are mostly negative this improves the
    interpretability of the solution
        theta: NxP matrix of theta estimates
        a: IxP matrix of a estimates

        returns: tuple of inverted theta and a paramters
    """
    totals = np.sum(a, axis=0)
    a *= totals / np.abs(totals)
    if theta is not None:
        theta *= totals / np.abs(totals)

    return a, theta

def MSE(est, true):
    """
    Mean square error
    Parameters
    ----------
    est: estimated parameter values
    true: true paremters values

    Returns
    -------
    the MSE
    """
    return np.mean(np.power(est-true,2))



# initialise model and optimizer
logger = CSVLogger("logs", name='pvae_movielens', version=0)
trainer = Trainer(fast_dev_run=False,
                  max_epochs=5000,
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=0.0000008, patience=100, mode='min')])

pvae = PartialVariationalAutoencoder(
                 nitems=3706,
                 emb_dim=32,
                 latent_dim=64,
                 hidden_layer_dim=32,
                 mirt_dim=3,
                 learning_rate=.005,
                 batch_size=100,
                 data_path='./data/movielens/ratings.dat',
                 beta=1)
trainer.fit(pvae)


# plot training loss
logs = pd.read_csv(f'logs/pvae_movielens/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/pvae_movielens/training_loss.png')

dataset = PVAE_Dataset('./data/movielens/ratings.dat')
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
user_ids, ratings, output = next(iter(dataloader))

pred = pvae(user_ids, ratings)

print(torch.sqrt(torch.mean((pred-output)**2)))






