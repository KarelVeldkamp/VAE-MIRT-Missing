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

with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']

# read data and true parameter values
data = pd.read_csv(f'./data/{cfg["which_data"]}/data.csv').iloc[:, 1:]
data = torch.tensor(data.values, dtype=torch.float32)
# read in true parameter estimates
a_true = pd.read_csv(f'./data/{cfg["which_data"]}/a.csv').iloc[:, 1:].values
theta_true = pd.read_csv(f'./data/{cfg["which_data"]}/theta.csv').iloc[:, 1:].values
d_true = pd.read_csv(f'./data/{cfg["which_data"]}/d.csv').iloc[:, 1:].values


# initialise model and optimizer
logger = CSVLogger("logs", name=cfg['which_data'], version=0)
trainer = Trainer(fast_dev_run=False,
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])
Q = a_true != 0
Q = Q.astype(int)
vae = VariationalAutoencoder(nitems=data.shape[1],
                             latent_dims=cfg['latent_dims'],
                             hidden_layer_size=int((data.shape[1]+2)*cfg['latent_dims']/2),
                             qm=Q,
                             learning_rate=cfg['learning_rate'],
                             batch_size=data.shape[0],#cfg['batch_size'],
                             data_path=f'./data/{cfg["which_data"]}/data.csv',
                             missing=True)
trainer.fit(vae)

a_est = vae.decoder.linear.weight.detach().numpy()[:, 0:3]
d_est = vae.decoder.linear.bias.detach().numpy()
missing = torch.isnan(data)
data[missing] = 0
mask = (~missing).int()
theta_est = vae.encoder.est_theta(data, mask).detach().numpy()
print(a_est.shape)
print(d_est.shape)
# invert factors for increased interpretability
a_est, theta_est = inv_factors(a_est, theta_est)


# plot training loss
logs = pd.read_csv(f'logs/{cfg["which_data"]}/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/{cfg["which_data"]}/training_loss.png')


# parameter estimation plot for a
for dim in range(3):
    plt.figure()
    ai_est = a_est[:,dim]
    ai_true = a_true[:,dim]
    mse = MSE(ai_est, ai_true)
    plt.scatter(y=ai_est, x=ai_true)
    plt.plot(ai_true, ai_true)
    #for i, x in enumerate(ai_true):
    #    plt.text(ai_true[i], ai_est[i], i)
    plt.title(f'Parameter estimation plot: a{dim+1}, MSE={round(mse,4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/{cfg["which_data"]}/param_est_plot_a{dim+1}.png')

    # parameter estimation plot for theta
    plt.figure()
    thetai_est = theta_est[:, dim]
    thetai_true = theta_true[:, dim]
    mse = MSE(thetai_est, thetai_true)
    plt.scatter(y=thetai_est, x=thetai_true)
    plt.plot(thetai_true, thetai_true)
    plt.title(f'Parameter estimation plot: theta{dim+1}, MSE={round(mse,4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/{cfg["which_data"]}/param_est_plot_theta{dim+1}.png')

# parameter estimation plot for d
plt.figure()
plt.scatter(y=d_est, x=d_true)
plt.plot(d_true, d_true)
mse = MSE(d_est, d_true)
plt.title(f'Parameter estimation plot: d, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/param_est_plot_d.png')



