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

# create data loader
#dataset = ResponseDataset('./data/data.csv')
#train_loader= DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
#N_items = dataset.x_train.shape[1]


# initialise model and optimizer
logger = CSVLogger("logs", name="my_logs")
trainer = Trainer(fast_dev_run=False,
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=1e-8, patience=100, mode='min')])
QMatrix = genfromtxt(cfg['Q_matrix_file'], delimiter=',')
vae = VariationalAutoencoder(latent_dims=cfg['latent_dims'],
                             hidden_layer_size=int((28+2)*cfg['latent_dims']/2),
                             qm=QMatrix,
                             learning_rate=cfg['learning_rate'],
                             batch_size=cfg['batch_size'],
                             data_path=cfg['data_path'])
trainer.fit(vae)
# ## train model
# loss_values = []
# for epoch in range(cfg['max_epochs']):
#     train_loss = train_epoch(vae,train_loader,optim)
#     loss_values.append(train_loss)
#     if epoch % 50 == 0:
#         print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, cfg['max_epochs'],train_loss))
#
# plt.title(f'Training loss')
# plt.plot(loss_values)
# plt.savefig(f'./figures/training_loss.png')
#
# save parameter estimates
data = pd.read_csv(cfg['data_path']).iloc[:, 1:]
data = torch.tensor(data.values, dtype=torch.float32)
a_est = vae.decoder.linear.weight.detach().numpy()
d_est = vae.decoder.linear.bias.detach().numpy()
theta_est = vae.encoder.est_theta(data).detach().numpy()


# invert factors for increased interpretability
a_est, theta_est = inv_factors(a_est, theta_est)

# read in true parameter estimates
a_true = np.loadtxt('./data/a.txt')
theta_true = np.loadtxt('./data/theta.txt')
d_true = np.genfromtxt('./data/d.txt')


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
    plt.savefig(f'./figures/param_est_plot_a{dim+1}.png')

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
    plt.savefig(f'./figures/param_est_plot_theta{dim+1}.png')

# parameter estimation plot for d
plt.figure()
plt.scatter(y=d_est, x=d_true)
plt.plot(d_true, d_true)
mse = MSE(d_est, d_true)
plt.title(f'Parameter estimation plot: d, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig('./figures/param_est_plot_d.png')



