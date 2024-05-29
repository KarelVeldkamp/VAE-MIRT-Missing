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


def cov2cor(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters:
    cov_matrix (np.ndarray): A square covariance matrix.

    Returns:
    np.ndarray: The corresponding correlation matrix.
    """
    # Compute the standard deviations
    stddev = np.sqrt(np.diag(cov_matrix))

    # Outer product of standard deviations
    outer_stddev = np.outer(stddev, stddev)

    # Create the correlation matrix
    cor_matrix = cov_matrix / outer_stddev

    # Fix any numerical issues that may have resulted in slightly off-diagonal values
    np.fill_diagonal(cor_matrix, 1)

    return cor_matrix