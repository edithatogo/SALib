import numpy as np
from SALib.analyze import fast, rbd_fast
import math
from scipy.stats import norm

# replicate previous bootstrap behaviour for comparison

def _bootstrap_replacement_fast(Y, M, resamples, conf_level):
    T_data = Y.shape[0]
    n_size = math.ceil(T_data * 0.5)
    res_S1 = np.zeros(resamples)
    res_ST = np.zeros(resamples)
    for _ in range(resamples):
        idx = np.random.choice(T_data, replace=True, size=n_size)
        Y_rs = Y[idx]
        N = len(Y_rs)
        omega = math.floor((N - 1) / (2 * M))
        S1, ST = fast.compute_orders(Y_rs, N, M, omega)
        res_S1[_] = S1
        res_ST[_] = ST
    bnd = norm.ppf(0.5 + conf_level / 2.0)
    return bnd * res_S1.std(ddof=1), bnd * res_ST.std(ddof=1)

def _bootstrap_replacement_rbd(X_d, Y, M, resamples, conf_level):
    T_data = X_d.shape[0]
    n_size = int(T_data * 0.5)
    res = np.zeros(resamples)
    for _ in range(resamples):
        idx = np.random.choice(T_data, replace=True, size=n_size)
        X_rs, Y_rs = X_d[idx], Y[idx]
        S1 = rbd_fast.compute_first_order(rbd_fast.permute_outputs(X_rs, Y_rs), M)
        S1 = rbd_fast.unskew_S1(S1, M, Y_rs.size)
        res[_] = S1
    return norm.ppf(0.5 + conf_level / 2.0) * res.std(ddof=1)


def test_fast_block_bootstrap_less_variance():
    Y = np.arange(100)
    np.random.seed(123)
    old_S1, old_ST = _bootstrap_replacement_fast(Y, 4, 50, 0.95)
    np.random.seed(123)
    new_S1, new_ST = fast.bootstrap(Y, 4, 50, 0.95)
    assert new_S1 <= old_S1
    assert new_ST <= old_ST


def test_rbd_fast_block_bootstrap_less_variance():
    X = np.linspace(0, 1, 100)
    Y = np.sin(2 * np.pi * X)
    np.random.seed(321)
    old_ci = _bootstrap_replacement_rbd(X, Y, 10, 50, 0.95)
    np.random.seed(321)
    new_ci = rbd_fast.bootstrap(X, Y, 10, 50, 0.95)
    assert new_ci <= old_ci
