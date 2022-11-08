from model import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from torch.utils.data import DataLoader


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


# create data loader
dataset = ResponseDataset('./data/data.csv')
train_loader= DataLoader(dataset, batch_size=10000, shuffle=False)

# initialise model and optimizer
QMatrix = genfromtxt('./QMatrix.csv', delimiter=',')
vae = VariationalAutoencoder(latent_dims=3, qm=QMatrix)
lr = 1e-3
optim = torch.optim.Adam(vae.parameters(), lr=lr)

# train model
num_epochs = 10000
loss_values = []
for epoch in range(num_epochs):
    train_loss = train_epoch(vae,train_loader,optim)
    loss_values.append(train_loss)
    if epoch % 50 == 0:
        print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, num_epochs,train_loss))

plt.title(f'Training loss')
plt.plot(loss_values)
plt.savefig(f'./figures/training_loss.png')

# save parameter estimates
a_est = vae.decoder.linear.weight.detach().numpy()
d_est = vae.decoder.linear.bias.detach().numpy()
theta_est = vae.encoder.est_theta(dataset.x_train).detach().numpy()


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



