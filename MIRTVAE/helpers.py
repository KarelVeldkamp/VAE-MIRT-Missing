## File with helper functions
from scipy.stats.stats import pearsonr
import numpy as np

def inv_factors(a_est, a_true, theta_est=None):
    """
    Helper function that inverts factors when discrimination values are mostly negative this improves the
    interpretability of the solution
        theta: NxP matrix of theta estimates
        a: IxP matrix of a estimates

        returns: tuple of inverted theta and a paramters
    """
    for dim in range(a_est.shape[1]):
        if pearsonr(a_est[:,dim], a_true[:,dim])[0] < 0:
            a_est[:, dim] *= -1
            theta_est[:, dim] *=-1

    return a_est, theta_est

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

def bias(est, true):
    return (est - true).mean()

def loglikelihood(a, d, theta, data):
    """
    Log likelihood for an MIRT model
    Parameters
    ----------
    a: np array of slopes
    d: np array of intercepts
    theta: np array of abilities
    data: np array of binary data

    Returns
    -------
    the log likelihood
    """
    exponent = np.matmul(theta, a.T) + d
    prob = np.exp(exponent) / (1 + np.exp(exponent))

    lll = np.sum(np.log(prob * data + (1-prob)*(1-data)))

    return lll

def Cor(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def cov2cor(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation