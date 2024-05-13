import torch.autograd

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
import sys
import time
import os
torch.set_num_threads(1)

# set working directory to source file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# set device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']

# overwrite configurations if command line arguments are provided
if len(sys.argv) > 1:
    cfg['N'] = int(sys.argv[1])
    cfg['missing_percentage'] = float(sys.argv[2])
    cfg["iteration"] = int(sys.argv[3])
    cfg['model'] = sys.argv[4]
    cfg['mirt_dim'] = int(sys.argv[5])

# simulate data
if cfg['simulate']:
    #theta=np.random.normal(0,1,cfg['N']*cfg['mirt_dim']).reshape((cfg['N'], cfg['mirt_dim']))
    mu_true = np.array([0, 0,0])
    cov_true = np.array([1, .0, .0,
                         .0, 1, .0,
                         .0, .0, 1]).reshape((3,3))
    theta = np.random.multivariate_normal(mu_true, cov_true, size=cfg['N'])

    Q = pd.read_csv(f'./QMatrices/QMatrix{cfg["mirt_dim"]}D.csv', header=None).values
    Q = np.zeros((60,3))
    Q[0:20, 0]=1
    Q[20:40, 1] = 1
    Q[40:60, 2] = 1


    a = np.random.uniform(.5, 2, Q.shape[0] * cfg['mirt_dim']).reshape((Q.shape[0], cfg['mirt_dim']))  # draw discrimination parameters from uniform distribution
    a *= Q
    b = np.linspace(-3, 3, Q.shape[0], endpoint=True)  # equally spaced values between -2 and 2 for the difficulty

    exponent = np.dot(theta, a.T) + b

    prob = np.exp(exponent) / (1 + np.exp(exponent))
    data = np.random.binomial(1, prob).astype(float)
else:
    #it = cfg["iteration"]
    it =1
    #data = pd.read_csv(f'./data/simulated/data_{cfg["mirt_dim"]}_{it}.csv', header=None, index_col=False).to_numpy()
    a = pd.read_csv(f'./parameters/simulated/a_{cfg["mirt_dim"]}_{it}.csv', header=None, index_col=False).to_numpy()
    b = np.squeeze(pd.read_csv(f'./parameters/simulated/b_{cfg["mirt_dim"]}_{it}.csv', header=None, index_col=False).to_numpy())
    theta = pd.read_csv(f'./parameters/simulated/theta_{cfg["mirt_dim"]}_{it}.csv', header=None, index_col=False).to_numpy()
    Q = pd.read_csv(f'./parameters/QMatrix{cfg["mirt_dim"]}D.csv', header=None).values

    exponent = np.dot(theta, a.T) + b
    prob = np.exp(exponent) / (1 + np.exp(exponent))
    data = np.random.binomial(1, prob).astype(float)

# potentially save data to disk
if cfg['save']:
    #np.savetxt(f'./data/simulated/data_{cfg["mirt_dim"]}_{cfg["iteration"]}.csv', data, delimiter=",")
    np.savetxt(f'./parameters/simulated/a_{cfg["mirt_dim"]}_{cfg["iteration"]}.csv', a, delimiter=",")
    np.savetxt(f'./parameters/simulated/b_{cfg["mirt_dim"]}_{cfg["iteration"]}.csv', b, delimiter=",")
    np.savetxt(f'./parameters/simulated/theta_{cfg["mirt_dim"]}_{cfg["iteration"]}.csv', theta, delimiter=",")

# introduce missingness
np.random.seed(cfg['iteration'])
indices = np.random.choice(data.shape[0]*data.shape[1], replace=False, size=int(data.shape[0]*data.shape[1]*cfg['missing_percentage']))
data[np.unravel_index(indices, data.shape)] = float('nan')

# Compute approximate latent covariance matrix
#Sigma_1 = torch.Tensor(np.corrcoef(np.dot(data, Q).T))
Sigma_1 = torch.Tensor(np.cov(theta.T))
Sigma_1 = torch.eye(3)


# make data tensor
data = torch.Tensor(data)



# X = pd.read_csv('./data/missing/data.csv', index_col=0).to_numpy()
# a = pd.read_csv('./data/missing/a.csv', index_col=0).to_numpy()
# theta = pd.read_csv('./data/missing/theta.csv', index_col=0).to_numpy()
# d = pd.read_csv('./data/missing/d.csv', index_col=0).to_numpy()
# Q = (a != 0).astype(int)

if os.path.exists('logs/simfit/version_0/metrics.csv'):
    os.remove('logs/simfit/version_0/metrics.csv')
# initialise model and optimizer
logger = CSVLogger("logs", name='simfit', version=0)



trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                  max_epochs=cfg['max_epochs'],
                  min_epochs=cfg['min_epochs'],
                  enable_checkpointing=False, 
                  logger=logger,
                  accelerator='cpu',
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')],
                  detect_anomaly=False,
                  gradient_clip_val=0)

if cfg['model'] == 'cvae':
    dataset = SimDataset(data)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
    vae = CVAE(nitems=data.shape[1],
               dataloader=train_loader,
               latent_dims=cfg['mirt_dim'],
               hidden_layer_size=cfg['hidden_layer_size'],
               qm=Q,
               learning_rate=cfg['learning_rate'],
               batch_size=cfg['batch_size'],
               beta=cfg['beta'],
               n_samples=cfg['n_iw_samples'],
               sigma_1=None,

    )
elif cfg['model'] == 'idvae':
    dataset = SimDataset(data)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
    vae = IDVAE(nitems=data.shape[1],
               dataloader=train_loader,
               latent_dims=cfg['mirt_dim'],
               hidden_layer_size=cfg['hidden_layer_size'],
               qm=Q,
               learning_rate=cfg['learning_rate'],
               batch_size=cfg['batch_size'],
               beta=cfg['beta'],
               n_samples=cfg['n_iw_samples']

    )
elif cfg['model'] == 'ivae':
    missing = torch.isnan(data)
    mask = (~missing).int()
    vae = IVAE(nitems=data.shape[1],
               dataloader=None,
               data=data,
               mask=mask,
               latent_dims=cfg['mirt_dim'],
               hidden_layer_size=cfg['hidden_layer_size'],
               qm=Q,
               learning_rate=cfg['learning_rate'],
               batch_size=cfg['batch_size'],
               beta=cfg['beta'],
               n_samples=cfg['n_iw_samples']
    )
elif cfg['model'] == 'pvae':
    dataset = PartialDataset(data)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
    vae = PVAE(nitems=Q.shape[0],
               dataloader=train_loader,
               latent_dims=1,
               hidden_layer_size=1,
               learning_rate=cfg['learning_rate'],
               batch_size=cfg['batch_size'],
               emb_dim=cfg['p_emb_dim'],
               h_hidden_dim=cfg['p_hidden_dim'],
               latent_dim=cfg['p_latent_dim'],
               hidden_layer_dim=cfg['p_hidden_layer_dim'],
               mirt_dim=cfg['mirt_dim'],
               qm=Q,
               beta=cfg['beta'],
               n_samples=cfg['n_iw_samples']
    )
elif cfg['model'] == 'vae':
    dataset = SimDataset(data)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    vae = VAE(nitems=data.shape[1],
               dataloader=train_loader,
               latent_dims=cfg['mirt_dim'],
               hidden_layer_size=cfg['hidden_layer_size'],
               qm=Q,
               learning_rate=cfg['learning_rate'],
               batch_size=cfg['batch_size'],
               beta=cfg['beta'],
               n_samples=cfg['n_iw_samples'],
               sigma_1 = None,
               cor_theta=cfg['cor_theta']

    )

else:
    raise Exception("Invalid model type")

start = time.time()
trainer.fit(vae)
runtime = time.time()-start
print(runtime)
a_est = vae.decoder.weights.t().detach().numpy()
d_est = vae.decoder.bias.t().detach().numpy()
#a_est = vae.decoder.linear.weight.detach().cpu().numpy()[:, 0:cfg['mirt_dim']]
#d_est = vae.decoder.linear.bias.detach().cpu().numpy()
vae = vae.to(device)

if cfg['model'] in ['cvae', 'iwae']:
    dataset = SimDataset(data, device)
    train_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)
    data, mask = next(iter(train_loader))
    theta_est, log_sigma_est  = vae.encoder(data, mask)
elif cfg['model'] in ['idvae', 'vae']:
    dataset = SimDataset(data, device)
    train_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)
    data, mask = next(iter(train_loader))
    theta_est, log_sigma_est = vae.encoder(data)
elif cfg['model'] == 'ivae':
    theta_est, log_sigma_est = vae.encoder(vae.data)
elif cfg['model'] == 'pvae':
    dataset = PartialDataset(data, device)
    train_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)
    item_ids, ratings, _, _ = next(iter(train_loader))
    theta_est, log_sigma_est = vae.encoder(item_ids, ratings)


mu = theta_est.unsqueeze(0)
sigma = log_sigma_est.unsqueeze(0)
_, _, theta_est = vae.sampler(mu, 0)
theta_est = theta_est.squeeze()



#sigma_est = torch.exp(log_sigma_est)
theta_est = theta_est.detach().cpu().numpy()
total_runtime = time.time()-start



sigma_est = log_sigma_est.detach().cpu().numpy()
print(f'total time: {time.time()-start}')
if cfg['mirt_dim'] == 1:
    theta = np.expand_dims(theta, 1)
# invert factors for increased interpretability
a_est, theta_est = inv_factors(a_est=a_est, theta_est=theta_est, a_true=a)


# W = vae.sampler.extract_cholesky_factor().detach().numpy()
#
# cov = W@W.T
# R = cov2cor(cov)
# print(cov)
# print(np.cov(theta_est.T))

#a_est = a_est @ np.diag(np.sqrt(np.diag(cov)))

print(np.corrcoef(theta_est.T))

mse_a = f'{MSE(a_est, a)}\n'
# bias_a = f'{np.mean(a_est-a)}\n'
# var_a = f'{np.var(a_est)}\n'
mse_d = f'{MSE(d_est, b)}\n'
# bias_d = f'{np.mean(d_est-b)}\n'
# var_d = f'{np.var(d_est)}\n'
mse_theta = f'{MSE(theta_est, theta)}\n'
# bias_theta = f'{np.mean(theta_est-theta)}\n'
# var_theta = f'{np.var(theta_est)}\n'

print(mse_a)
print(mse_d)
print(mse_theta)


# if hasattr(vae.transform, 'L'):
#     L = torch.zeros((3, 3))  # create empty matrix for L
#     L[0, 0] = 1
#     triu_indices = torch.triu_indices(3, 3, offset=1)
#     L[triu_indices[0], triu_indices[1]] = F.tanh(vae.transform.L)
#
#     L = L / (torch.square(L).sum(0).sqrt() + 1e4)  # scale so that sum of diagonal elements < 1
#
#     L = torch.add(L, torch.diag_embed(torch.sqrt(1 - torch.square(L).sum(0))))
#
#     ### Compute cholesky factor of covariance matrix
#     # R = torch.matrix_exp(L) # matrix exponent to ensure cov matrix is positive definite
#
#     ### compute correlation matrix
#     R = L.t() @ L
#
#     print(R)



#
#
# lll = f'{loglikelihood(a_est, d_est, theta_est, data.numpy())}\n'
# runtime = f'{runtime}\n'
# ms = f'{np.mean(sigma_est)}\n'
# ss = f'{np.std(sigma_est)}\n'
#
# # When run with command line arguments, save results to file
# if len(sys.argv) > 1:
#     with open(f"../results/{'_'.join(sys.argv[1:])}.txt", 'w') as f:
#         f.writelines([mse_a, mse_d, mse_theta, lll, runtime, ms, ss, bias_a, bias_d, bias_theta, var_a, var_d, var_theta])
if len(sys.argv) > 1:
    par_names = ['theta', 'a', 'd']
    par = []
    value = []
    par_i = []
    par_j = []
    for i, est in enumerate([theta_est, a_est, np.expand_dims(d_est, 1)]):
        for r in range(est.shape[0]):
            for c in range(est.shape[1]):
                par.append(par_names[i])
                value.append(est[r, c])
                par_i.append(r)
                par_j.append(c)

    result = pd.DataFrame({'n': cfg['N'], 'missing': cfg['missing_percentage'], 'iteration': cfg['iteration'],
                           'model': cfg['model'], 'mirt_dim': cfg['mirt_dim'], 'parameter': par, 'i':par_i, 'j':par_j, 'value': value})

    #result.to_csv(f"../results/{'_'.join(sys.argv[1:])}.csv", index=False)
    with open(f"../results/{'_'.join(sys.argv[1:])}.txt", 'w') as f:
        f.write('%.5f' % total_runtime)

# otherwise, print results and plot figures
else:
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

    #if cfg['mirt_dim'] ==1:
    #     a = np.expand_dims(a, 1)
    #     theta = np.expand_dims(theta, 1)
    # parameter estimation plot for a
    for dim in range(cfg['mirt_dim']):
        plt.figure()

        ai_est = a_est[:,dim]
        ai_true = a[:,dim]

        mse = MSE(ai_est, ai_true)

        plt.scatter(y=ai_est, x=ai_true)
        plt.plot(ai_true, ai_true)
        slope, intercept = np.polyfit(ai_true, ai_est, 1)  # Get slope and intercept of the regression line
        plt.plot(ai_true, slope * np.array(ai_true) + intercept, '--', label='Regression Line', alpha=.3)  # Plot the regression line
        #for i, x in enumerate(ai_true):
        #    plt.text(ai_true[i], ai_est[i], i)
        plt.title(f'Parameter estimation plot: a{dim+1}, MSE={round(mse,4)}')
        plt.xlabel('True values')
        plt.ylabel('Estimates')
        plt.savefig(f'./figures/simfit/param_est_plot_a{dim+1}.png')

        # parameter estimation plot for theta
        plt.figure()
        thetai_est = theta_est[:, dim].squeeze()
        thetai_true = theta[:, dim].squeeze()
        mse = MSE(thetai_est, thetai_true)
        plt.scatter(y=thetai_est, x=thetai_true)
        plt.plot(thetai_true, thetai_true)
        slope, intercept = np.polyfit(thetai_true, thetai_est, 1)  # Get slope and intercept of the regression line
        plt.plot(thetai_true, slope * np.array(thetai_true) + intercept, '--',label='Regression Line', alpha=.3)  # Plot the regression line
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




