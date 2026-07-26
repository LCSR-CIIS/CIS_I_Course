import numpy as np

def sample_normal(mean, cov, n=1):
    '''
    input: mean - mean vector of the multivariate normal distribution
           cov - covariance matrix of the multivariate normal distribution
           n - number of samples to generate (default is 1)
    output: samples - list of sampled vectors from the multivariate normal distribution
    '''
    return np.random.multivariate_normal(mean, cov, n)