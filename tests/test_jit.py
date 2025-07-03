import numpy as np
from numpy.testing import assert_allclose

from SALib.sample.morris.morris import _generate_trajectory
from SALib.analyze.delta import calc_delta
from SALib.analyze.fast import bootstrap as fast_bootstrap
from SALib.analyze.rbd_fast import bootstrap as rbd_fast_bootstrap


def test_generate_trajectory_jit_consistency():
    if not hasattr(_generate_trajectory, "py_func"):
        return
    G = np.array([[1, 0], [0, 1], [0, 1]])
    res_jit = _generate_trajectory(G, 4)
    assert res_jit.shape == (3, 3)
    assert np.all((res_jit >= 0) & (res_jit <= 1))


def test_calc_delta_jit_consistency():
    if not hasattr(calc_delta, "py_func"):
        return
    rng = np.random.default_rng(1)
    Y = rng.random(100)
    X = rng.random(100)
    Ygrid = np.linspace(Y.min(), Y.max(), 10)
    m = np.array([0, 50, 100])
    res_jit = calc_delta(Y, Ygrid, X, m)
    res_py = calc_delta.py_func(Y, Ygrid, X, m)
    assert_allclose(res_jit, res_py)


def test_fast_bootstrap_jit_consistency():
    if not hasattr(fast_bootstrap, "py_func"):
        return
    rng = np.random.default_rng(2)
    Y = rng.random(100)
    np.random.seed(2)
    res_jit = fast_bootstrap(Y, 4, 5, 0.95)
    np.random.seed(2)
    res_py = fast_bootstrap.py_func(Y, 4, 5, 0.95)
    assert_allclose(res_jit, res_py)


def test_rbd_fast_bootstrap_jit_consistency():
    if not hasattr(rbd_fast_bootstrap, "py_func"):
        return
    rng = np.random.default_rng(3)
    X = rng.random((100, 2))
    Y = rng.random(100)
    np.random.seed(3)
    res_jit = rbd_fast_bootstrap(X, Y, 10, 5, 0.95)
    np.random.seed(3)
    res_py = rbd_fast_bootstrap.py_func(X, Y, 10, 5, 0.95)
    assert_allclose(res_jit, res_py)
