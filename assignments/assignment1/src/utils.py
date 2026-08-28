import numpy as np

from data_types.nominal_types import vct3


def sample_normal(mean, cov, n=1):
    '''
    input: mean - mean vector of the multivariate normal distribution, OR a
               single vct3, OR a List[vct3] (one marker/point per entry)
           cov - covariance matrix of the multivariate normal distribution;
               if mean is a List[vct3] of length N and cov is 3x3, the same
               3x3 covariance is applied i.i.d. to each point (block-diagonal
               3N x 3N); a 3N x 3N cov is also accepted directly
           n - number of samples to generate (default is 1)
    output: samples - if mean was a single vct3: a vct3 (n=1) or List[vct3]
               (n>1); if mean was a List[vct3]: a List[vct3] (n=1) or
               List[List[vct3]] (n>1); otherwise, the original ndarray
               behavior of np.random.multivariate_normal
    '''
    single = isinstance(mean, vct3)
    mean_list = [mean] if single else mean

    if isinstance(mean_list, (list, tuple)) and len(mean_list) > 0 and isinstance(mean_list[0], vct3):
        n_points = len(mean_list)
        flat_mean = np.concatenate([m.vec.ravel() for m in mean_list])
        cov_arr = np.asarray(cov, dtype=np.float64)
        if cov_arr.shape == (3, 3) and n_points > 1:
            cov_arr = np.kron(np.eye(n_points), cov_arr)
        samples = np.random.multivariate_normal(flat_mean, cov_arr, n)  # (n, 3*n_points)
        out = [[vct3(row[3 * i:3 * i + 3]) for i in range(n_points)] for row in samples]
        if single:
            out = [row[0] for row in out]
        if n == 1:
            return out[0]
        return out

    return np.random.multivariate_normal(mean, cov, n)