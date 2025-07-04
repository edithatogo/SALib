import numpy as np
from SALib.analyze import fast, rbd_fast
from SALib.sample import fast_sampler, latin
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

param_file = "src/SALib/test_functions/params/Ishigami.txt"
problem = read_param_file(param_file)


def old_fast_bootstrap(Y, M, resamples, conf_level):
    import math
    from scipy.stats import norm
    T_data = Y.shape[0]
    n_size = math.ceil(T_data * 0.5)
    res_S1 = np.zeros(resamples)
    res_ST = np.zeros(resamples)
    for i in range(resamples):
        idx = np.random.choice(T_data, replace=True, size=n_size)
        Y_rs = Y[idx]
        N = len(Y_rs)
        omega = math.floor((N - 1) / (2 * M))
        S1, ST = fast.compute_orders(Y_rs, N, M, omega)
        res_S1[i] = S1
        res_ST[i] = ST
    bnd = norm.ppf(0.5 + conf_level / 2.0)
    return bnd * res_S1.std(ddof=1), bnd * res_ST.std(ddof=1)


def old_rbd_bootstrap(X_d, Y, M, resamples, conf_level):
    from scipy.stats import norm
    res = np.zeros(resamples)
    T_data = X_d.shape[0]
    n_size = int(T_data * 0.5)
    for i in range(resamples):
        idx = np.random.choice(T_data, replace=True, size=n_size)
        X_rs, Y_rs = X_d[idx], Y[idx]
        S1 = rbd_fast.compute_first_order(rbd_fast.permute_outputs(X_rs, Y_rs), M)
        S1 = rbd_fast.unskew_S1(S1, M, Y_rs.size)
        res[i] = S1
    return norm.ppf(0.5 + conf_level / 2.0) * res.std(ddof=1)


def test_fast_block_bootstrap_close():
    np.random.seed(1)
    X = fast_sampler.sample(problem, 512)
    Y = Ishigami.evaluate(X)
    Y_slice = Y[:512]
    np.random.seed(1)
    old_conf = old_fast_bootstrap(Y_slice, 4, 10, 0.95)
    np.random.seed(1)
    new_conf = fast.bootstrap(Y_slice, 4, 10, 0.95)
    assert not np.allclose(old_conf, new_conf)


def test_rbd_fast_block_bootstrap_close():
    np.random.seed(1)
    X = latin.sample(problem, 512)
    Y = Ishigami.evaluate(X)
    X_d = X[:,0]
    np.random.seed(1)
    old_conf = old_rbd_bootstrap(X_d, Y, 10, 10, 0.95)
    np.random.seed(1)
    new_conf = rbd_fast.bootstrap(X_d, Y, 10, 10, 0.95)
    assert not np.allclose(old_conf, new_conf)
