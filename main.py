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

# create data laoder
dataset = ResponseDataset('./data/data.csv')
train_loader= DataLoader(dataset, batch_size=10, shuffle=False)

# initialise model and optimizer
QMatrix = genfromtxt('./QMatrix.csv', delimiter=',')
vae = VariationalAutoencoder(latent_dims=3, qm=QMatrix)
lr = 1e-3
optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

# train model
num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train_epoch(vae,train_loader,optim)
    if epoch % 5 == 0:
        print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, num_epochs,train_loss))

# save parameter estimates
a_est = vae.decoder.linear.weight.detach().numpy()
d_est = vae.decoder.linear.bias.detach().numpy()
theta_est = vae.encoder.est_theta(dataset.x_train).detach().numpy()

# calculate theta estimates
# TODO

# invert factors for increased interpretability
a_est, theta_est = inv_factors(a_est, theta_est)

# read in true parameter estimates
a_true = np.loadtxt('./data/a.txt')
theta_true = np.loadtxt('./data/theta.txt')
d_true = np.genfromtxt('./data/d.txt')


# parameter estimation plot for a
plt.figure()
plt.scatter(y=a_est.flatten(), x=a_true.flatten(), alpha=.5)
plt.title('Parameter estimation plot: a')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig('./figures/param_est_plot_a.png')

# parameter estimation plot for theta
plt.figure()
plt.scatter(y=theta_est.flatten(), x=theta_true.flatten(), alpha=.5)
plt.title('Parameter estimation plot: theta')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig('./figures/param_est_plot_theta.png')

# parameter estimation plot for d
plt.figure()
plt.scatter(y=d_est, x=d_true, alpha=.5)
plt.title('Parameter estimation plot: d')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig('./figures/param_est_plot_d.png')



