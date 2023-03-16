from model import *
from data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
ndim = a_true.shape[1]
theta_true = pd.read_csv(f'./data/{cfg["which_data"]}/theta.csv').iloc[:, 1:].values
d_true = pd.read_csv(f'./data/{cfg["which_data"]}/d.csv').iloc[:, 1:].values

# introduce missingness
indices = np.random.choice(data.shape[0]*data.shape[1], replace=False, size=int(data.shape[0]*data.shape[1]*cfg['missing_percentage']))
data[np.unravel_index(indices, data.shape)] = float('nan')

# # Sample parameter values
# a_true=np.random.uniform(.5,2,cfg['nitems']*cfg['mirt_dim']).reshape((cfg['nitems'],cfg['mirt_dim']))      # draw discrimination parameters from uniform distribution
# if cfg['mirt_dim'] == 1:
#     a_true[0,] = 0
# elif cfg['mirt_dim'] == 2:
#     a_true[0:5, 0] = 0
#     a_true[6:10, 1] = 0
# elif cfg['mirt_dim'] == 3:
#     a_true[0:10, 0] = 0
#     a_true[11:20, 1] = 0
#     a_true[21:30, 2] = 0
#
# Q = (a_true != 0).astype(int)
#
# theta_true=np.sort(np.random.normal(0,1,cfg['N']*cfg['mirt_dim']).reshape((cfg['N'], cfg['mirt_dim'])))
# d_true=np.linspace(-2,2,cfg['nitems'],endpoint=True)   # equally spaced values between -3 and 3 for the difficulty
#
# # simulate data
# exponent = np.dot(theta_true, a_true.T)+d_true
# prob = np.exp(exponent)/(1+np.exp(exponent))
# data = np.random.binomial(1, prob).astype(float)
#
#
#
# # introduce missingness
# indices = np.random.choice(data.shape[0]*data.shape[1], replace=False, size=int(data.shape[0]*data.shape[1]*cfg['missing_percentage']))
# data[np.unravel_index(indices, data.shape)] = float('nan')
# data = torch.Tensor(data)

# initialise model and optimizer
logger = CSVLogger("logs", name=cfg['which_data'], version=0)
trainer = Trainer(fast_dev_run=False,
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])
Q = a_true != 0
Q = Q.astype(int)

dataset = PartialDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
vae = PVAE(dataloader=train_loader,
           nitems=cfg['nitems'],
           learning_rate=cfg['learning_rate'],
           batch_size=data.shape[0],
           emb_dim=cfg['p_emb_dim'],
           h_hidden_dim=cfg['p_hidden_dim'],
           latent_dim=cfg['p_latent_dim'],
           hidden_layer_dim=cfg['p_hidden_layer_dim'],
           mirt_dim=cfg['mirt_dim'],
           Q=Q,
           beta=1)
trainer.fit(vae)

a_est = vae.decoder.linear.weight.detach().numpy()[:, 0:3]
d_est = vae.decoder.linear.bias.detach().numpy()
missing = torch.isnan(torch.Tensor(data))
data[missing] = 0
mask = (~missing).int()
theta_est, _ = vae.encoder(data)#, mask)
theta_est = theta_est.detach().numpy()
# invert factors for increased interpretability
a_est, theta_est = inv_factors(a_est, theta_est)

print(MSE(a_est, a_true))
print(MSE(d_est, d_true))
print(MSE(theta_est, theta_true))


# plot training loss
logs = pd.read_csv(f'logs/{cfg["which_data"]}/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/{cfg["which_data"]}/training_loss.png')
# # plot binary cross entropy
# plt.clf()
# plt.plot(logs['epoch'], logs['binary_cross_entropy'])
# plt.title('Binary Cross Entropy')
# plt.savefig(f'./figures/{cfg["which_data"]}/binary_cross_entropy.png')
# # plot KL divergence
# plt.clf()
# plt.plot(logs['epoch'], logs['kl_divergence'])
# plt.title('KL Divergence')
# plt.savefig(f'./figures/{cfg["which_data"]}/kl_divergence.png')


# parameter estimation plot for a
for dim in range(ndim):
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



