from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset


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

# Sample parameter values
a=np.random.uniform(.5,3,cfg['nitems']*cfg['mirt_dim']).reshape((cfg['nitems'],cfg['mirt_dim']))      # draw discrimination parameters from uniform distribution
if cfg['mirt_dim'] == 1:
    a[0,] = 0
elif cfg['mirt_dim'] == 2:
    a[0:5, 0] = 0
    a[6:10, 1] = 0
elif cfg['mirt_dim'] == 3:
    a[0:10, 0] = 0
    a[11:20, 1] = 0
    a[21:30, 2] = 0

Q = (a != 0).astype(int)

theta=np.sort(np.random.normal(0,1,cfg['N']*cfg['mirt_dim']).reshape((cfg['N'], cfg['mirt_dim'])))
b=np.linspace(-3,3,cfg['nitems'],endpoint=True)   # eqally spaced values between -3 and 3 for the difficulty

# simulate data
exponent = np.dot(theta,a.T)+b
prob = np.exp(exponent)/(1+np.exp(exponent))
X = np.random.binomial(1, prob).astype(float)

# introduce missingness
s_miss=np.random.randint(cfg['N'],size=int(np.floor(cfg['missing_percentage']*cfg['N']*cfg['nitems'])))
i_miss=np.random.randint(cfg['nitems'],size=int(np.floor(cfg['missing_percentage']*cfg['N']*cfg['nitems'])))
X[s_miss,i_miss]=float('nan')

# X = pd.read_csv('./data/missing/data.csv', index_col=0).to_numpy()
# a = pd.read_csv('./data/missing/a.csv', index_col=0).to_numpy()
# theta = pd.read_csv('./data/missing/theta.csv', index_col=0).to_numpy()
# d = pd.read_csv('./data/missing/d.csv', index_col=0).to_numpy()
# Q = (a != 0).astype(int)


# initialise model and optimizer
logger = CSVLogger("logs", name='simfit', version=0)
trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])

if cfg['model'] == 'cvae':
    dataset = SimDataset(X)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
    vae = CVAE(nitems=X.shape[1],
               dataloader=train_loader,
               latent_dims=cfg['mirt_dim'],
               hidden_layer_size=cfg['hidden_layer_size'],
               hidden_layer_size2=cfg['hidden_layer_size2'],
               hidden_layer_size3=cfg['hidden_layer_size3'],
               qm=Q,
               learning_rate=cfg['learning_rate'],
               batch_size=X.shape[0]

    )
if cfg['model'] == 'ivae':

    vae = IVAE(nitems=X.shape[1],
               data=X,
               latent_dims=cfg['mirt_dim'],
               hidden_layer_size=cfg['hidden_layer_size'],
               hidden_layer_size2=cfg['hidden_layer_size2'],
               qm=Q,
               learning_rate=cfg['learning_rate'],
               batch_size=X.shape[0],#cfg['batch_size'],
               s_miss=s_miss,
               i_miss=i_miss,
    )
if cfg['model'] == 'pvae':
    dataset = PartialDataset(X)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
    vae = PVAE(dataloader=train_loader,
               nitems=cfg['nitems'],
               learning_rate=cfg['learning_rate'],
               batch_size=X.shape[0],
               emb_dim=cfg['p_emb_dim'],
               h_hidden_dim=cfg['p_hidden_dim'],
               latent_dim=cfg['p_latent_dim'],
               hidden_layer_dim=cfg['p_hidden_layer_dim'],
               mirt_dim=cfg['mirt_dim'],
               Q=Q,
               beta=1
    )


trainer.fit(vae)

a_est = vae.decoder.linear.weight.detach().cpu().numpy()[:, 0:cfg['mirt_dim']]
d_est = vae.decoder.linear.bias.detach().cpu().numpy()

if cfg['model'] == 'cvae':
    data, mask = next(iter(train_loader))
    theta_est, _ = vae.encoder(data, mask)
    theta_est = theta_est.detach().cpu().numpy()
elif cfg['model'] == 'ivae':
    theta_est, _ = vae.encoder(vae.data)
    theta_est = theta_est.detach().cpu().numpy()
elif cfg['model'] == 'pvae':
    item_ids, ratings, _, _ = next(iter(train_loader))
    theta_est, _ = vae.encoder(item_ids, ratings)
    theta_est = theta_est.detach().cpu().numpy()
# invert factors for increased interpretability
a_est, theta_est = inv_factors(a_est, theta_est)

print(MSE(a_est, a))
print(MSE(d_est, b))
print(MSE(theta_est, theta))

# plot training loss
logs = pd.read_csv(f'logs/simfit/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/simfit/training_loss.png')
# plot binary cross entropy
# plt.clf()
# plt.plot(logs['epoch'], logs['binary_cross_entropy'])
# plt.title('Binary Cross Entropy')
# plt.savefig(f'./figures/simfit/binary_cross_entropy.png')
# # plot KL divergence
# plt.clf()
# plt.plot(logs['epoch'], logs['kl_divergence'])
# plt.title('KL Divergence')
# plt.savefig(f'./figures/simfit/kl_divergence.png')

if cfg['mirt_dim'] ==1:
     a = np.expand_dims(a, 1)
     theta = np.expand_dims(theta, 1)
# parameter estimation plot for a
for dim in range(cfg['mirt_dim']):
    plt.figure()

    ai_est = a_est[:,dim]
    ai_true = a[:,dim]

    mse = MSE(ai_est, ai_true)
    plt.scatter(y=ai_est, x=ai_true)
    plt.plot(ai_true, ai_true)
    #for i, x in enumerate(ai_true):
    #    plt.text(ai_true[i], ai_est[i], i)
    plt.title(f'Parameter estimation plot: a{dim+1}, MSE={round(mse,4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/simfit/param_est_plot_a{dim+1}.png')

    # parameter estimation plot for theta
    plt.figure()
    thetai_est = theta_est[:, dim]
    thetai_true = theta[:, dim]
    mse = MSE(thetai_est, thetai_true)
    plt.scatter(y=thetai_est, x=thetai_true)
    plt.plot(thetai_true, thetai_true)
    plt.title(f'Parameter estimation plot: theta{dim+1}, MSE={round(mse,4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/simfit/param_est_plot_theta{dim+1}.png')

# parameter estimation plot for d
plt.figure()
plt.scatter(y=d_est, x=b)
plt.plot(b,b)
mse = MSE(d_est, b)
plt.title(f'Parameter estimation plot: d, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/simfit/param_est_plot_d.png')



